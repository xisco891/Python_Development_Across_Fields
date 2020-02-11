
import socket
import re
from functools import wraps


class ValidationRules():


    def isValidEmail(self, value):
        try:
            if self.is_empty_or_none(value) == True:
                return False

            match = re.match('^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$', value)

            if match is None:
                return False
            return True
        except:
            return False


    def is_valid_mdm_status(self, value):
        try:
            if self.is_empty_or_none(value) == True:
                return False

            if value in ['installing', 'online', 'offline', 'archived', 'deleted']:
                return True

            return False

        except:
            return False


    def is_valid_ipv4_address(self, address):
        if self.is_empty_or_none(address) == True:
            return False

        try:
            socket.inet_pton(socket.AF_INET, address)
        except AttributeError:  # no inet_pton here, sorry
            try:
                socket.inet_aton(address)
            except socket.error:
                return False
            return address.count('.') == 3
        except socket.error:  # not a valid address
            return False

        return True


    def is_valid_ipv6_address(self, address):
        if self.is_empty_or_none(address) == True:
            return False

        try:
            socket.inet_pton(socket.AF_INET6, address)
        except socket.error:  # not a valid address
            return False
        return True


    def is_valid_os_info(self, os_info_version):
        if self.is_empty_or_none(os_info_version) == True:
            return False

        os_array = os_info_version.split('.')
        if len(os_array) < 8:
            return False

        return True



validation = ValidationRules()
