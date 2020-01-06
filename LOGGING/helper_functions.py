
from validation_rules import validation
import requests, json
from IkLogger import CMyIkLogger

def createLoginSession(self, username=None, password=None, operatorName='Ikarus', partnerName=None):
    try:
        user = username
        pwd = password
        operator = operatorName

        if validation.is_empty_or_none(user) == True or validation.is_empty_or_none(pwd) == True:
            user = self.username
            pwd = self.password

        if validation.is_empty_or_none(operator) == True:
            operator = self.operatorName

        data_json = {"username": str(user), "password": str(pwd)}
        headers = {"Content-type": "application/json"}

        response = requests.post(self.loginSessionUrl, json=data_json, headers=headers)
        if response is None:
            return None

        if response.status_code == 200:
            json_response_text = json.loads(response.text)
            if json_response_text is None:
                CMyIkLogger.warning("Error creating 7P - API loginSession! response.text is None!")
                return None

            if 'id' in json_response_text == False:
                CMyIkLogger.warning("Error creating 7P - API loginSession! response has no X-CM-SID!")
                return None

            self.loginSession['id'] = json_response_text['id']

            if 'role' in json_response_text == False:
                CMyIkLogger.warning("Error creating 7P - API loginSession! response has no role!")
            else:
                self.loginSession['role'] = json_response_text['role']

            if 'role_id' in json_response_text == False:
                CMyIkLogger.warning("Error creating 7P - API loginSession! response has no role_id!")
            else:
                self.loginSession['role_id'] = json_response_text['role_id']

            if 'started' in json_response_text == False:
                CMyIkLogger.warning("Error creating 7P - API loginSession! response has no started!")
            else:
                self.loginSession['started'] = json_response_text['started']

            self.username = user
            self.password = pwd

            if 'role' in self.loginSession and self.loginSession['role'].lower() == 'operator':
                if 'role_id' in self.loginSession and validation.is_int(self.loginSession['role_id'], True) == True:
                    if self.getOperatorByID(int(self.loginSession['role_id'])) is None:
                        return None
            else:
                if not (operator is None):
                    if self.getOperator(operator) is None:
                        return None

            if not (partnerName is None):
                if self.getPartnerByName(partnerName) is None:
                    return None

            return True

        else:
            CMyIkLogger.warning(
                "Error creating 7P - API loginSession! status_code:" + str(response.status_code) + "; text: " + str(
                    response.text))
            return None

    except Exception as Ex:
        CMyIkLogger.warning("Exception! Error creating 7P - API loginSession! " + str(Ex))
        return None