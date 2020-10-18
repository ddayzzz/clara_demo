"""The client of the federated training process"""

import logging
import socket
import threading
import time
import math
import horovod.tensorflow as hvd
from multiprocessing.dummy import Pool as ThreadPool

import grpc
from mpi4py import MPI

comm = MPI.COMM_WORLD

import fed_learn.protos.federated_pb2 as fed_msg
import fed_learn.protos.federated_pb2_grpc as fed_service
from fed_learn.client.client_model_manager import ClientModelManager
from fed_learn.fed_utils import FED_DELTA_W, VERBOSE


class FederatedClient:
    """
    Federated client-side implementation
    """

    def __init__(self,
                 client_id,
                 client_args,
                 secure_train,
                 server_args=None,
                 exclude_vars=None,
                 privacy=None,
                 retry_timeout=30,
                 data_assembler=None):
        self.logger = logging.getLogger('FederatedClient')
        self.verbose = VERBOSE

        self.uid = client_id
        self.token = None
        self.client_args = client_args
        self.servers = server_args
        self.model_manager = ClientModelManager(
            task_names=tuple(self.servers),
            exclude_vars=exclude_vars,
            privacy=privacy,
        )

        self.pool = ThreadPool(len(self.servers))
        self.should_stop = False
        self.train_end = False
        self.retry = int(math.ceil(float(retry_timeout)/5))

        self.secure_train = secure_train
        self.data_assembler = data_assembler

    def client_registration(self, task_name):
        """
        Client's meta data used to authenticate and communicate.

        :return: a ClientLogin message.
        """
        local_ip = socket.gethostbyname(socket.gethostname())
        login_message = fed_msg.ClientLogin(
            client_id=self.uid, token=self.token, client_ip=local_ip)
        login_message.meta.task.name = task_name
        return login_message

    def client_state(self, task_name):
        """
        Client's meta data used to authenticate and communicate.

        :return: a ClientState message.
        """
        state_message = fed_msg.ClientState(uid=self.uid, token=self.token)
        state_message.meta.task.name = task_name
        return state_message

    def set_up_channel(self, channel_dict):
        """
        Connect client to the server.

        :param channel_dict: grpc channel parameters
        :return: an initialised grpc channel
        """
        if self.secure_train:
            with open(self.client_args['ssl_root_cert'], 'rb') as f:
                trusted_certs = f.read()
            with open(self.client_args['ssl_private_key'], 'rb') as f:
                private_key = f.read()
            with open(self.client_args['ssl_cert'], 'rb') as f:
                certificate_chain = f.read()

            credentials = grpc.ssl_channel_credentials(
                certificate_chain=certificate_chain,
                private_key=private_key,
                root_certificates=trusted_certs)

            # make sure that all headers are in lowecase,
            # otherwise grpc throws an exception
            call_credentials = grpc.metadata_call_credentials(
                lambda context, callback: callback((("x-custom-token", self.
                                                     token), ), None))
            # use this if you want standard "Authorization" header
            # call_credentials = grpc.access_token_call_credentials(
            #     "x-custom-token")
            composite_credentials = grpc.composite_channel_credentials(
                credentials, call_credentials)
            channel = grpc.secure_channel(
                **channel_dict, credentials=composite_credentials)

        else:
            channel = grpc.insecure_channel(**channel_dict)
        return channel

    def grpc_error_handler(self, grpc_error, verbose=False) -> bool:
        """
        Handling grpc exceptions
        """
        decrease_retry = True
        status_code = grpc_error.code()
        if grpc.StatusCode.UNAVAILABLE == status_code:
            self.logger.info(
                'Could not connect to server. '
                'Setting flag for stopping training. (%s)',
                grpc_error.details())
            self.should_stop = True

        if grpc.StatusCode.RESOURCE_EXHAUSTED == status_code:
            if grpc_error.details().startswith('No token'):
                self.logger.info(
                    'No token for this client in current round. '
                    'Waiting for server new round. '
                )
                self.should_stop = False
                decrease_retry = False
        if verbose:
            self.logger.debug(grpc_error)
        return decrease_retry

    def fetch_remote_model(self, task_name):
        """
        Get registered with the remote server via channel,
        and fetch the server's model parameters.

        :param task_name: server identifier string
        :return: a CurrentModel message from server
        """
        reg_result, m_result, retry = None, None, self.retry
        with self.set_up_channel(self.servers[task_name]) as channel:
            stub = fed_service.FederatedTrainingStub(channel)
            if not self.token:
                while retry > 0:
                    try:
                        reg_result = stub.Register(
                            self.client_registration(task_name))
                        # self.logger.info('Registration: {}'.format(reg_result))
                        self.token = reg_result.token
                        self.logger.info('Successfully registered client:{} for {}. Got token:{}'.
                                         format(self.uid, task_name, self.token))
                        # Clear the stopping flag
                        # if the connection to server recovered.
                        self.should_stop = False
                        break
                    except grpc.RpcError as grpc_error:
                        self.grpc_error_handler(grpc_error, verbose=self.verbose)
                        retry -= 1
                        time.sleep(5)
                if self.should_stop:
                    self.train_end = True
                if reg_result is None:
                    return None

            retry = self.retry
            while retry > 0:
                # get the global model
                try:
                    m_result = stub.GetModel(self.client_state(task_name))
                    # Clear the stopping flag
                    # if the connection to server recovered.
                    self.should_stop = False

                    self.logger.info(
                        'Received {} model at round {} ({} Bytes), update signal: {}'.format(task_name,
                        m_result.meta.current_round, m_result.ByteSize(), m_result.allowed_to_perform_update))
                    break
                except grpc.RpcError as grpc_error:
                    decrease_retry = self.grpc_error_handler(grpc_error, verbose=self.verbose)
                    if decrease_retry:
                        retry -= 1
                    time.sleep(5)

                    # self.logger.info('Retry fetching model...({})'.format(
                    #       server_info))
            if self.should_stop:
                self.train_end = True
        return m_result

    def push_remote_model(self, task_name):
        """
        Read local model and push to self.server[task_name] channel.
        This function makes and sends a Contribution Message.

        :param task_name: should be one of the keys of `self.server`
        """
        # contrib = fed_msg.Contribution()
        # # set client auth. data
        # contrib.client.CopyFrom(self.client_state(task_name))
        # # set model meta info.
        # model_meta = self.model_manager.model_meta(task_name)
        # contrib.client.meta.CopyFrom(model_meta)
        # # set num. of local iterations
        # contrib.n_iter = self.model_manager.num_local_iterations()
        # # set contribution type
        # contrib.type = FED_DELTA_W
        # # set model data
        # model_data = self.model_manager.read_current_model(
        #     task_name, contrib.type)
        # contrib.data.CopyFrom(model_data)

        client_state = self.client_state(task_name)
        contrib = self.data_assembler.get_contribution_data(self.model_manager, task_name, client_state)



        server_msg, retry = None, self.retry
        with self.set_up_channel(self.servers[task_name]) as channel:
            stub = fed_service.FederatedTrainingStub(channel)
            while retry > 0:
                try:
                    self.logger.info('Send %s at round %s', task_name,
                                     contrib.client.meta.current_round)
                    server_msg = stub.SubmitUpdate(contrib)
                    # Clear the stopping flag
                    # if the connection to server recovered.
                    self.should_stop = False

                    self.logger.info('Received comments: %s %s',
                                     server_msg.meta.task.name,
                                     server_msg.comment)
                    break
                except grpc.RpcError as grpc_error:
                    if grpc_error.details().startswith('Contrib'):
                        self.logger.info('Publish model failed: %s', grpc_error.details())
                        break  # outdated contribution, no need to retry
                    self.grpc_error_handler(grpc_error, verbose=self.verbose)
                    retry -= 1
                    time.sleep(5)
        return server_msg

    def push_remote_fake_update(self, task_name):
        """
        Read local model and push to self.server[task_name] channel.
        This function makes and sends a Contribution Message.

        :param task_name: should be one of the keys of `self.server`
        """

        state_message = fed_msg.ClientState(uid=self.uid, token=self.token)
        state_message.meta.task.name = task_name
        # 组装信息
        contrib = fed_msg.Contribution()
        # set client auth. data
        contrib.client.CopyFrom(state_message)
        # 服务需要验证 ModelMeta 信息
        model_meta = self.model_manager.model_meta(task_name)
        contrib.client.meta.CopyFrom(model_meta)



        server_msg, retry = None, self.retry
        with self.set_up_channel(self.servers[task_name]) as channel:
            stub = fed_service.FederatedTrainingStub(channel)
            while retry > 0:
                try:
                    self.logger.info('Send fake update data')
                    server_msg = stub.SubmitUpdate(contrib)
                    # Clear the stopping flag
                    # if the connection to server recovered.
                    self.should_stop = False

                    self.logger.info('Received comments: %s %s',
                                     server_msg.meta.task.name,
                                     server_msg.comment)
                    break
                except grpc.RpcError as grpc_error:
                    if grpc_error.details().startswith('Contrib'):
                        self.logger.info('Publish fake model failed: %s', grpc_error.details())
                        break  # outdated contribution, no need to retry
                    self.grpc_error_handler(grpc_error, verbose=self.verbose)
                    retry -= 1
                    time.sleep(5)
        return server_msg

    def send_heartbeat(self, task_name):
        if self.token:
            token = fed_msg.Token()
            token.token = self.token

            with self.set_up_channel(self.servers[task_name]) as channel:
                stub = fed_service.FederatedTrainingStub(channel)
                try:
                    self.logger.debug('Send %s heartbeat %s', task_name, self.token)
                    stub.Heartbeat(token)
                except grpc.RpcError as grpc_error:
                    pass
                    # self.grpc_error_handler(grpc_error, verbose=self.verbose)
                    # self.train_end = True

    # 以下的方法封装了整个联邦学习客户端的流程
    def heartbeat(self):
        """
        Sends a heartbeat from the client to the server.
        """
        return self.pool.map(self.send_heartbeat, tuple(self.servers))

    def pull_models(self):
        """
        Fetch remote models and update the local client's session.
        """
        remote_models = self.pool.map(self.fetch_remote_model, tuple(self.servers))
        # TODO: if some of the servers failed
        # send_fake = all([model.allowed_to_perform_update == 2 or model.allowed_to_perform_update == 0 for model in remote_models])
        send_fake = all(
            [model.allowed_to_perform_update == 2 for model in remote_models])
        return self.model_manager.assign_current_model(remote_models), send_fake

    def push_models(self, push_fake=False):
        """
        Push the local model to multiple servers.
        """
        if push_fake:
            return self.pool.map(self.push_remote_fake_update, tuple(self.servers))
        else:
            return self.pool.map(self.push_remote_model, tuple(self.servers))

    def federated_step(self, fitter):
        """
        Run a federated step.
        """
        # self.model_manager.set_fitter(fitter)
        client_local_rank = fitter.train_ctx.my_rank
        if client_local_rank == 0:
            # 获取的数据都是列表的形式
            pull_success, send_fake = self.pull_models()
        else:
            pull_success, send_fake = False, False

        if fitter.multi_gpu:

            assert fitter.multi_gpu, "目前不支持单机使用的 update signal"
            pull_success = comm.bcast(pull_success, root=0)
            self.train_end = comm.bcast(self.train_end, root=0)

        if pull_success:  # pull models from multiple servers
            if fitter.multi_gpu:
                hvd.broadcast_global_variables(root_rank=0)
            if self.model_manager.train():  # do local fitting
                if client_local_rank == 0:
                    self.push_models()  # push models
        elif send_fake:
            if client_local_rank == 0:
                self.push_models(push_fake=True)  # push models




    def run_federated_steps(self, fitter):
        """
        Keep running federated steps.
        """
        self.model_manager.set_fitter(fitter)

        while not self.train_end:
            self.federated_step(fitter)

    def run_heartbeat(self):
        """
        Periodically runs the heartbeat.
        """
        while not self.train_end:
            self.heartbeat()
            time.sleep(60)

    def run(self, fitter):
        """
        Run the client-side as a thread
        """
        # thread = threading.Thread(
        #     target=self.run_federated_steps, args=[fitter])
        # thread.daemon = True
        # thread.start()

        heartbeat_thread = threading.Thread(target=self.run_heartbeat)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()

        try:
            self.run_federated_steps(fitter)

        # except KeyboardInterrupt:
        #     pass
        finally:
            self.train_end = True
            return self.close()

    def quit_remote(self, task_name):
        """
        Sending the last message to the server before leaving.

        :param task_name: server task identifier
        :return: server's reply to the last message
        """
        server_message, retry = None, 3
        with self.set_up_channel(self.servers[task_name]) as channel:
            stub = fed_service.FederatedTrainingStub(channel)
            while retry > 0:
                try:
                    self.logger.info('Quitting server: %s', task_name)
                    server_message = stub.Quit(self.client_state(task_name))
                    # Clear the stopping flag
                    # if the connection to server recovered.
                    self.should_stop = False

                    self.logger.info('Received comment from server: %s',
                                     server_message.comment)
                    break
                except grpc.RpcError as grpc_error:
                    self.grpc_error_handler(grpc_error)
                    retry -= 1
                    time.sleep(3)
        return server_message

    def close(self):
        """
        Quit the remote federated server, close the local session.
        """
        self.logger.info('Shutting down client')
        self.should_stop = True

        self.pool.map(self.quit_remote, tuple(self.servers))
        try:
            self.pool.terminate()  # clean up all threads
        except Exception as e:
            self.logger.info(e)
        # pool = ThreadPool(len(self.servers))
        # pool.map(self.quit_remote, tuple(self.servers))

        # TF sometimes holds the thread not being able to shutdown cleanly. Not call this for now.
        # self.model_manager.close()

        # import traceback, sys
        # for thread in threading.enumerate():
        #     traceback.print_stack(sys._current_frames()[thread.ident])

        return 0
