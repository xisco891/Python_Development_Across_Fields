# -*- coding: utf-8 -*-

import configparser
from validation_rules import validation
from globals import globals

###INCLUDE THIS INTO A DECORATOR
# if self._isLoaded and 'CONFIG' in self._config

class ConfigFile():
    """
    This class contains the configuration file and handles all config data.
    After importing it is possible to use config to get access to all of the methods without initializing it more than
    once.
    """
    # holds after loading config the config data
    _config = configparser.ConfigParser()
    # state that defines if a config is loaded
    _isLoaded = False

    def __init__(self, type=None):
        """
        Loads data from config. If you import this module somewhere else it is only initialized once
        see https://docs.python.org/3.4/tutorial/modules.html#more-on-modules
        """

        try:
            self.load(type)
        except ValueError as err_info:
            print("ERROR: Failed to open or read from config file.")
            raise err_info

    def get_database_path(self):
        """
        Estimate current database path that is set within config
        :return: database path
        """
        if self._isLoaded and 'CONFIG' in self._config and 'dbpath' in self._config['CONFIG']:
            return self._config['CONFIG']['dbpath']

    def get_logserver(self):
        """
        :return: logserver path
        """
        if self._isLoaded and 'CONFIG' in self._config and 'logserver' in self._config['CONFIG']:
            return self._config['CONFIG']['logserver']
        return None

    def get_database_path_sqlAuth(self):
        """
        Estimate current database path that is set within config
        :return: database path
        """
        if self._isLoaded and 'CONFIG' in self._config and 'dbpath_sqlAuth' in self._config['CONFIG']:
            return self._config['CONFIG']['dbpath_sqlAuth']

    def get_allowedhosts(self):
        if self._isLoaded and 'CONFIG' in self._config and 'allowedhosts' in self._config['CONFIG']:
            return self._config['CONFIG']['allowedhosts']

        return None

    def get_version_file_fullpath(self):
        if self._isLoaded and 'CONFIG' in self._config and 'version_file_fullpath' in self._config['CONFIG']:
            return self._config['CONFIG']['version_file_fullpath']

        return None

    def get_webserver(self):
        if self._isLoaded and 'CONFIG' in self._config and 'webserver' in self._config['CONFIG']:
            return self._config['CONFIG']['webserver']

        return None

    def get_clientPrefix(self, default='avAPI-'):
        if self._isLoaded and 'API' in self._config and 'clientPrefix' in self._config['API']:
            if self._config['API']['clientPrefix'] is None or len(self._config['API']['clientPrefix']) == 0:
                return default
            return self._config['API']['clientPrefix']

        return default

    def get_licenseApiUrl(self):
        if self._isLoaded and 'API' in self._config and 'license_api' in self._config['API']:
            if self._config['API']['license_api'] is None or len(self._config['API']['license_api']) == 0:
                return None
            return self._config['API']['license_api']

        return None

    def get_static_license_path(self):
        try:
            if self._isLoaded and 'CONFIG' in self._config and 'static_license_path' in self._config['CONFIG']:
                return self._config['CONFIG']['static_license_path']
            return None
        except:
            return None

    def use_static_license(self):
        try:
            from data_provider.helper import helper
            if self._isLoaded and 'CONFIG' in self._config and 'use_static_license' in self._config['CONFIG']:
                return helper.to_bool(self._config['CONFIG']['use_static_license'], False)
        except:
            return False

    def get_crm_api_server_url(self, default_api_url='http://admin.isc.local/cgi-bin_internal/api.pl'):
        try:
            if self._isLoaded and 'API' in self._config and 'crm_api_server_url' in self._config['API']:
                return self._config['API']['crm_api_server_url']
            return default_api_url
        except:
            return default_api_url

    def getDefaultSmtpServer(self):
        try:
            if self._isLoaded and 'CONFIG' in self._config and 'default_smtp_server' in self._config['CONFIG']:
                return self._config['CONFIG']['default_smtp_server']
            return None
        except:
            return None

    def get_default_smtp_recipient(self, default='fink.s@ikarus.at'):
        try:
            if self._isLoaded and 'CONFIG' in self._config and 'default_smtp_recipient' in self._config['CONFIG']:
                if validation.isValidEmail(self._config['CONFIG']['default_smtp_recipient']) == True:
                    return self._config['CONFIG']['default_smtp_recipient']
            return default
        except:
            return default

    def get_7p_api_username(self):
        try:
            if self._isLoaded and 'API' in self._config and 'api_username_7p' in self._config['API']:
                return self._config['API']['api_username_7p']
            return None
        except:
            return None

    def get_7p_api_password(self):
        try:
            if self._isLoaded and 'API' in self._config and 'api_password_7p' in self._config['API']:
                return self._config['API']['api_password_7p']
            return None
        except:
            return None

    def get_7p_api_server_base_url(self, default_api_url='https://mdm.ikarus.at/cloudmanagement/data/'):
        try:
            if self._isLoaded and 'API' in self._config and 'api_server_base_url_7p' in self._config['API']:
                return self._config['API']['api_server_base_url_7p']
            return default_api_url
        except:
            return default_api_url

    def get_7p_user_login_server_base_url(self, default_api_url='https://mdm.ikarus.at/'):
        try:
            if self._isLoaded and 'API' in self._config and 'get_7p_user_login_server_base_url' in self._config['API']:
                return self._config['API']['get_7p_user_login_server_base_url']
            return default_api_url
        except:
            return default_api_url

    def log_request_response_to_db(self):
        try:
            from data_provider.helper import helper
            if self._isLoaded and 'CONFIG' in self._config and 'log_request_response_to_db' in self._config['CONFIG']:
                return helper.to_bool(self._config['CONFIG']['log_request_response_to_db'], False)
        except:
            return False

    def get_log_request(self):
        try:
            from data_provider.helper import helper
            if self._isLoaded and 'CONFIG' in self._config and 'log_request' in self._config['CONFIG']:
                return helper.to_bool(self._config['CONFIG']['log_request'], False)
        except:
            return False

    def get_log_response(self):
        try:
            from data_provider.helper import helper
            if self._isLoaded and 'CONFIG' in self._config and 'log_response' in self._config['CONFIG']:
                return helper.to_bool(self._config['CONFIG']['log_response'], False)
            return False
        except:
            return False

    def get_default_push_api_host(self):
        try:
            if self._isLoaded and 'API' in self._config and 'default_push_api_host' in self._config['API']:
                return self._config['API']['default_push_api_host']
            return None
        except:
            return None

    def get_default_push_api_certificatePath(self):
        try:
            if self._isLoaded and 'API' in self._config and 'default_push_api_certificatePath' in self._config['API']:
                return self._config['API']['default_push_api_certificatePath']
            return None
        except:
            return None

    def get_default_push_api_pemPath(self):
        try:
            if self._isLoaded and 'API' in self._config and 'default_push_api_pemPath' in self._config['API']:
                return self._config['API']['default_push_api_pemPath']
            return None
        except:
            return None

    def get_default_push_api_keyPath(self):
        try:
            if self._isLoaded and 'API' in self._config and 'default_push_api_keyPath' in self._config['API']:
                return self._config['API']['default_push_api_keyPath']
            return None
        except:
            return None

    def get_default_push_api_certificatePassword(self):
        try:
            if self._isLoaded and 'API' in self._config and 'default_push_api_certificatePassword' in self._config['API']:
                return self._config['API']['default_push_api_certificatePassword']
            return None
        except:
            return None

    def get_load_balancer(self):
        if self._isLoaded and 'CONFIG' in self._config and 'load_balancer' in self._config['CONFIG']:
            return self._config['CONFIG']['load_balancer']

    def get_ssl_credentials(self):
        ssl_credentials = ['ssl_version','ca_certs','keyfile','certfile','cert_reqs', 'server_side']
        if self._isLoaded and 'CONFIG' in self._config and (x in self._config['CONFIG'] for x in ssl_credentials):
            return { x : self._config['CONFIG'][x] for x in ssl_credentials }

    def get_credentials(self):
        if self._isLoaded and 'CONFIG' in self._config and 'user' and 'password' in self._config['CONFIG']:
            return self._config['CONFIG']['user'], self._config['CONFIG']['password'] 

    def use_push_api(self):
        try:
            from data_provider.helper import helper
            if self._isLoaded and 'API' in self._config and 'use_default_push_api' in self._config['API']:
                return helper.to_bool(self._config['API']['use_default_push_api'], False)
            return False
        except:
            return False

    def load(self, type):
        """
        Loads config file to get the needed information
        :return: true if loading config was successful otherwise false
        """
        result=None
        if type is None:
            result = self._config.read( globals.CONFIGFILE)
        if type == 'client':
            result = self._config.read(globals.CLIENT_CONFIGFILE)
        if type == 'server':
            result = self._config.read(globals.SERVER_CONFIGFILE)

        if len(result) == 0:
            text = "ERROR: Failed to open or read from config file! "
            raise ValueError(text)
            self._isLoaded = False
            return False

        self._isLoaded = True
        return True

    def is_config_loaded(self):
        return self._isLoaded

class ClientConfig(ConfigFile):
    def __init__(self):
        super(ClientConfig, self).__init__('client')

class ServerConfig(ConfigFile):
    def __init__(self):
        super(ClientConfig, self).__init__('server')

# import this config to all other modules to use config members
#ServerConfig = ServerConfig()
#ClientConfig = ClientConfig()
config = ConfigFile()
