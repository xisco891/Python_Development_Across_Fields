

# from data_provider.IKLogger import CMyIkLogger
#
# from rabbitMQ.python.src.action.ExclusionAction import
# from rabbitMQ.python.src.action.InfectionAction import
# from rabbitMQ.python.src.action.PingPongAction import
# from rabbitMQ.python.src.action.ScanAction import
#
# from rabbitMQ.python.src.cloud.DeviceRequest_pb2 import
# from rabbitMQ.python.src.cloud.DeviceResponse_pb2 import
# from rabbitMQ.python.src.cloud.VersionUpdateRequest import
#
# from rabbitMQ.python.src.cloud.data.CloudHeader_pb2 import
# from rabbitMQ.python.src.cloud.data.DeviceInfo_pb2 import
# from rabbitMQ.python.src.cloud.data.LicenseInfo_pb2 import
# from rabbitMQ.python.src.cloud.data.VersionData_pb2 import
#
# from rabbitMQ.python.src.common.ActionType_pb2 import
# from rabbitMQ.python.src.common.CoreActions_pb2 import
#
# from rabbitMQ.python.src.communication.Command_pb2 import
# from rabbitMQ.python.src.communication.Event_pb2 import
# from rabbitMQ.python.src.communication.ResponseExternal_pb2 import
#
# from rabbitMQ.python.src.core.Exclusion_pb2 import
# from rabbitMQ.python.src.core.Infection_pb2 import
# from rabbitMQ.python.src.core.PingPong_pb2 import
# from rabbitMQ.python.src.core.Scan_pb2 import
#
# from rabbitMQ.python.src.header.ExternalHeader_pb2 import
# from rabbitMQ.python.src.header.InternalHeader_pb2 import
#
#
# if __name__ == "__main__":
#     return True


class Protocol_Buffer:

    def __init__(self):
        self.message = protobuf_pb2.protocol()

    def read_event(self,body):
        try:
            self.message.deviceid = "deviceid from body"
            self.message.protocol = "protobuf"
            self.message.api_request.date_request = "command string extracted from body"
            self.message.settings.value = "value extracted from body"
            self.message.settings.subtype = "subtype extracted from body"
            return self.message

        except Exception as Ex:
            return False

    def read_return(self,body):

        try:
            self.message.jobid = "job.id extracted from body"
            self.message.date_response = "xx.xx.xxxx - date extracted from the body"
            return True

        except Exception as Ex:
            return False

    def send_request(self,body):

        try:
            self.command_to_avi.id = body.id
            self.command_to_avi.cmd = body.cmd
            return True

        except Exception as Ex:
            return False