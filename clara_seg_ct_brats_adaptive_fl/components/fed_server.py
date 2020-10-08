"""federated server for aggregating and sharing federated model"""

import logging
import uuid
import time
from concurrent import futures
from threading import Condition, Lock

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

import fed_learn.protos.federated_pb2 as fed_msg
import fed_learn.protos.federated_pb2_grpc as fed_service
from fed_learn.fed_utils import GRPC_DEFAULT_OPTIONS
from fed_learn.server.server_model_manager import ServerModelManager


class FederatedServer(fed_service.FederatedTrainingServicer):
    """
    Federated model aggregation services
    """

    def __init__(self,
                 task_name=None,
                 min_num_clients=2,
                 max_num_clients=10,
                 wait_after_min_clients=10,
                 start_round=1,
                 num_rounds=-1,
                 exclude_vars=None,
                 model_log_dir=None,
                 ckpt_preload_path=None,
                 model_aggregator=None,
                 heart_beat_timeout=600):
        """

        :param start_round: 0 indicates init. the global model randomly.
        :param min_num_clients: minimum number of contributors at each round.
        :param max_num_clients: maximum number of contributors at each round.
        """
        assert model_log_dir is not None, 'model_log_dir must be specified to save checkpoint'

        self.logger = logging.getLogger('FederatedServer')

        self.task_name = task_name  # should uniquely define a tlt workflow
        self.min_num_clients = max(min_num_clients, 1)
        # self.max_num_clients = min(max_num_clients, self.min_num_clients)
        self.max_num_clients = max(max_num_clients, 1)
        self.model_manager = ServerModelManager(
            start_round=start_round,
            num_rounds=num_rounds,
            exclude_vars=exclude_vars,
            model_log_dir = model_log_dir,
            ckpt_preload_path=ckpt_preload_path,
            model_aggregator=model_aggregator,
            keep_round_model=True
        )

        self.auth_client_id = {}  # {'client_1', 'client_2', 'client_3'}
        self.grpc_server = None
        self.lock = Lock()
        self.sync = Condition()
        self.accumulator = list()  # accumulating client's contributions
        self.tokens = None
        self.update_error = False
        self.round_started = Timestamp()
        self.round_started.GetCurrentTime()
        with self.lock:
            self.reset_tokens()

        self.wait_after_min_clients = wait_after_min_clients
        self.heart_beat_timeout = heart_beat_timeout

    @property
    def current_round(self):
        """
        the current number of federated global step
        :return:
        """
        return self.model_manager.current_round

    @property
    def model_meta_info(self):
        """
        the model_meta_info uniquely defines the current model,
        it is used to reject outdated client's update.

        :return: model meta data object
        """
        if self.should_stop:
            return None
        meta_info = fed_msg.ModelMetaData()
        meta_info.created.CopyFrom(self.round_started)
        meta_info.current_round = self.current_round
        meta_info.task.name = self.task_name
        return meta_info

    @property
    def should_stop(self):
        """

        :return: True to stop the main thread
        """
        return self.model_manager.should_stop

    def reset_tokens(self):
        """
        restart the token set, so that each client can take a token
        and start fetching the current global model.
        This function is not thread-safe.
        """
        last_time = self.round_started.seconds
        self.round_started.GetCurrentTime()
        now_time = self.round_started.seconds
        reset_duration = now_time - last_time
        self.logger.info(
            'Round time: %s second(s).',
            reset_duration if reset_duration > 0 else 'less than a')

        self.tokens = dict()
        for client in self.auth_client_id.keys():
            self.tokens[client] = self.model_meta_info

    def Register(self, request, context):
        """
        register new clients on the fly.
        Each client must get registered before getting the global model.
        The server will expect updates from the registered clients
        for multiple federated rounds.

        This function does not change min_num_clients and max_num_clients.
        """
        token = self.login_client(request, context)
        client_ip = context.peer().split(':')[1]
        # allow model requests/updates from this client

        if self.is_from_authorized_client(token):
            # previously known client, potentially contributed already
            # will join the next round
            return fed_msg.FederatedSummary(
                comment='Already registered', token=token)

        if len(self.auth_client_id) >= self.max_num_clients:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                          'Maximum number of clients reached')

        # new client will join the current round immediately
        self.auth_client_id.update({token: time.time()})
        self.tokens[token] = self.model_meta_info
        self.logger.info(
            'Client: New client {} joined. Sent token: {}.  Total clients: {}'.
            format(request.client_id + '@' + client_ip, token,
                   len(self.auth_client_id)))
        return fed_msg.FederatedSummary(
            comment='New client registered', token=token)

    def Quit(self, request, context):
        """
        existing client quits the federated training process.
        Server will stop sharing the global model with the client,
        further contribution will be rejected.

        This function does not change min_num_clients and max_num_clients.
        """
        token = self.validate_client(request, context)
        self.auth_client_id.pop(token)
        self.tokens.pop(token, None)
        self.logger.info('Client: {} left.  Total clients: {}'.format(
            token, len(self.auth_client_id)))
        return fed_msg.FederatedSummary(comment='Removed client')

    def GetModel(self, request, context):
        """
        process client's request of the current global model
        """
        token = self.validate_client(request, context)
        if token is None:
            return None

        model_meta_info = self.tokens.pop(token, None)
        if model_meta_info is None:
            # no more tokens for this round
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                          'No token available for the current round')
        # with self.sync:
        #     if not self.tokens:
        #         self.sync.notify_all()
        #     else:
        #         self.sync.wait()

        with self.lock:
            model = fed_msg.CurrentModel()
            model.meta.CopyFrom(model_meta_info)

            # fetch current global model
            model.data.CopyFrom(self.model_manager.model)
            return model

    def SubmitUpdate(self, request, context):
        """
        handling client's submission of the federated updates
        running aggregation if there are enough updates
        """
        self.update_error = False
        token = self.validate_client(request.client, context)

        if token is None:
            response_comment = 'Ignored the submit from invalid client. '
            self.logger.info(response_comment)

        # if len(self.accumulator) > self.min_num_clients:
        #     context.abort(grpc.StatusCode.ALREADY_EXISTS,
        #                   'Contrib: already enough in the current round')
        else:

            model_meta = self.is_valid_contribution(request.client.meta)
            if model_meta is None:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                              'Contrib: invalid for the current round')
                response_comment = 'Invalid contribution. '
                self.logger.info(response_comment)
            else:

                client_contrib_id = '{}_{}_{}'.format(model_meta.task.name, token,
                                                      model_meta.current_round)

                start_time = request.client.meta.created
                timenow = Timestamp()
                timenow.GetCurrentTime()
                time_seconds = timenow.seconds - start_time.seconds
                self.logger.info(
                    'received %s (%s Bytes, %s seconds)', client_contrib_id,
                    request.ByteSize(), time_seconds or 'less than 1')

                if self.save_contribution(client_contrib_id, request):
                    with self.lock:
                        self.accumulator.append(request)
                        # if self.get_enough_updates():
                        #     self.aggregate()
                        num_of_updates = len(self.accumulator)

                    # Only the first one meets the minimum clients trigger the aggregation.
                    if num_of_updates == self.min_num_clients:
                        if num_of_updates < len(self.auth_client_id):
                            self.logger.debug("Starting to wait. {}".format(self.wait_after_min_clients))
                            time.sleep(self.wait_after_min_clients)
                        self.aggregate()

                response_comment = \
                    'Received round {} from {} ({} Bytes, {} seconds)'.format(
                        request.client.meta.current_round, request.client.uid,
                        request.ByteSize(), time_seconds or 'less than 1')

        summary_info = fed_msg.FederatedSummary(comment=response_comment)
        if self.model_meta_info is not None:
            summary_info.meta.CopyFrom(self.model_meta_info)
        return summary_info

    # def get_enough_updates(self):
    #     if len(self.accumulator) >= self.min_num_clients:
    #         if len(self.accumulator) >= len(self.auth_client_id):
    #             return True
    #         if self.round_stopping is None:
    #             self.round_stopping = time.time()
    #     else:
    #         return False
    #     #
    #     # return len(self.accumulator) >= self.min_num_clients

    def Heartbeat(self, request, context):
        token = request.token
        self.auth_client_id.update({token: time.time()})
        self.logger.debug('Receive heartbeat from Client:{}'.format(token))
        summary_info = fed_msg.FederatedSummary()
        return summary_info

    def save_contribution(self, client_contrib_id, data):
        """
        save the client's current contribution.

        :return: True iff it is successfully saved
        """
        return True

    def aggregate(self):
        """
        invoke model aggregation using the accumulator's content,
        then reset the tokens and accumulator.

        :return:
        """
        self.logger.info('> aggregating: %s', self.current_round)
        self.model_manager.aggregate(self.accumulator)
        self.reset_tokens()
        self.accumulator.clear()
        self.round_stopping = None

    def login_client(self, client_login, context):
        """
        validate the client state message

        :param client_state: A ClientState message received by server
        :param context: gRPC connection context
        :param allow_new: whether to allow new client. Its task should
            still match server's.
        :return: client id if it's a valid client
        """
        if not self.is_valid_task(client_login.meta.task):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                'Requested task does not match the current server task')
        if not self.is_authenticated_client(client_login):
            context.abort(grpc.StatusCode.UNAUTHENTICATED,
                          'Unknown client identity')

        # Return and use the existing token.
        if client_login.token:
            return client_login.token

        token = str(uuid.uuid4())
        return token

    def validate_client(self, client_state, context, allow_new=False):
        """
        validate the client state message

        :param client_state: A ClientState message received by server
        :param context: gRPC connection context
        :param allow_new: whether to allow new client. Its task should
            still match server's.
        :return: client id if it's a valid client
        """
        token = client_state.token
        if not token:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          'Could not read client uid from the payload')
            token = None
        if not self.is_valid_task(client_state.meta.task):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                'Requested task does not match the current server task')
            token = None
        if not (allow_new or self.is_from_authorized_client(token)):
            context.abort(grpc.StatusCode.UNAUTHENTICATED,
                          'Unknown client identity')
            token = None

        return token

    def is_authenticated_client(self, client_login):
        """

        :param client_login:
        :return:
        """
        # Use SSL certificate for authenticate the client. Return true here.
        return True

    def is_from_authorized_client(self, client_id):
        """
        simple authentication of the client.

        :return: True indicates it is a recognised client
        """
        return client_id in self.auth_client_id

    def is_valid_task(self, task):
        """
        check whether the requested task matches the server's task
        """
        return task.name == self.task_name

    def is_valid_contribution(self, contrib_meta_data):
        """
        check if the client submitted a valid contribution
        contribution meta should be for the current task and
        for the current round; matching server's model meta data.

        :param contrib_meta_data: Contribution message's meta data
        :return: the meta data if the contrib's meta data is valid,
            None otherwise.
        """
        server_meta = self.model_meta_info
        if '{}'.format(contrib_meta_data) == '{}'.format(server_meta):
            return contrib_meta_data
        return None

    def close(self):
        """
        shutdown the server.

        :return:
        """
        self.logger.info('shutting down server')
        try:
            if self.model_manager:
                self.model_manager.close()
        except RuntimeError:
            self.logger.info('closing model manager')
        try:
            if self.lock:
                self.lock.release()
            if self.sync:
                self.sync.release()
        except RuntimeError:
            self.logger.info('canceling sync locks')
        try:
            if self.grpc_server:
                self.grpc_server.stop(0)
        finally:
            self.logger.info('server off')
            return 0

    def deploy(self, grpc_args=None, secure_train=False):
        """
        start a grpc server and listening the designated port.
        """
        num_server_workers = grpc_args.get('num_server_workers', 1)
        num_server_workers = max(self.min_num_clients, num_server_workers)
        target = grpc_args['service'].get('target', '0.0.0.0:6007')
        grpc_options = grpc_args['service'].get('options',
                                                GRPC_DEFAULT_OPTIONS)

        if not self.grpc_server:
            self.grpc_server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=num_server_workers),
                options=grpc_options)
            fed_service.add_FederatedTrainingServicer_to_server(
                self, self.grpc_server)

        if secure_train:
            with open(grpc_args['ssl_private_key'], 'rb') as f:
                private_key = f.read()
            with open(grpc_args['ssl_cert'], 'rb') as f:
                certificate_chain = f.read()
            with open(grpc_args['ssl_root_cert'], 'rb') as f:
                root_ca = f.read()

            server_credentials = grpc.ssl_server_credentials(
                ((
                    private_key,
                    certificate_chain,
                ), ),
                root_certificates=root_ca,
                require_client_auth=True)
            self.grpc_server.add_secure_port(target, server_credentials)
            self.logger.info('starting secure server at %s', target)
        else:
            self.grpc_server.add_insecure_port(target)
            self.logger.info('starting insecure server at %s', target)
        self.grpc_server.start()

        try:
            while not self.should_stop:
                # Clean and remove the dead client without heartbeat.
                delete = []
                for token, value in self.auth_client_id.items():
                    if value < time.time() - self.heart_beat_timeout:
                        delete.append(token)

                for token in delete:
                    self.auth_client_id.pop(token)
                    self.tokens.pop(token, None)
                    self.logger.info('Remove the dead Client: {}.  Total clients: {}'.format(
                            token, len(self.auth_client_id)))

                time.sleep(3)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(e)
        finally:
            return self.close()
