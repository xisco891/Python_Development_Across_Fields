# -*- coding: utf-8 -*-
import logging


def constant(f):
    def fset(self, value):
        raise SyntaxError

    def fget(self):
        return f(self)

    return property(fget, fset)
  

class _Constants(object):
    @constant
    def PORT(self):
        return 5235

    @constant
    def PORT_TEST(self):
        return 5236

    # Define logging level
    @constant
    def LOGGING(self):
        return logging.INFO

    @constant
    def CONFIGFILE(self):
        return "c:/etc/avitc_api.conf"

    @constant
    def SERVER_CONFIGFILE(self):
        return "c:/etc/avitc_server-api.conf"

    @constant
    def CLIENT_CONFIGFILE(self):
        return "c:/etc/avitc_client-api.conf"

    @constant
    def LOGPATH(self):
        return "logs"

    @constant
    def DEFAULT_SCAN_PROFILES(self):
        return ["FULL", "STANDARD", "QUICK", "REMOVEABLE"]

    @constant
    def FULL_LOGPATH(self):
        return self.BASEPATH + '/' + self.LOGPATH

    @constant
    def LOGNAME(self):
        return "avic_python.log"

    # Every x seconds to check if new entries in T_MESSAGE_PUSH table are available
    @constant
    def DB_CHECK_FOR_UPDATES(self):
        return 10

    # database status for setting something to "pending"
    @constant
    def DB_STATUS_PENDING(self):
        return "PENDING"

    # database status for setting something to "sent"
    @constant
    def DB_STATUS_SENT(self):
        return "SENT"

    # database status for setting something to "sent_ok"
    @constant
    def DB_STATUS_SENT_OK(self):
        return "SENT_OK"

    # database status for setting something to "device_received"
    @constant
    def DB_STATUS_RECEIVED(self):
        return "RECEIVED"

    # database status for setting something to "failed"
    @constant
    def DB_STATUS_FAIL(self):
        return "FAILED"

    # database status for setting something to "failed"
    @constant
    def DB_STATUS_RECEIVED_FAILED(self):
        return "RECEIVED_FAILED"

    # number of pending tasks that should be send
    @constant
    def DB_POLLING_PENDING(self):
        return 20

    # Define path where config file is available
    @constant
    def BASEPATH(self):
        #return "C:\\mmwgui\\GCM-python"
        #return "c:\\mobile.security_gcm"
        return "."

globals = _Constants()