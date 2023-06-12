import sys
sys.path.append(".")

import pytest
from requests.models import Response

from chemspace.pug_utils import regulate_api_requests

class TestPUGUtils:
    @pytest.mark.parametrize("headers,wait_time",[
        ("Request Count status: Green (0%), Request Time status: Green (0%), Service status: Green (20%)", 0.2),
        ("Request Count status: Green (0%), Request Time status: Green (0%), Service status: Black (99%)", 3600.0),
        ("Request Count status: Green (0%), Request Time status: Red (80%), Service status: Green (20%)", 60.0),
        ("Request Count status: Yellow (40%), Request Time status: Green (0%), Service status: Green (20%)", 1.0),
        ])
    def test_rest_regulation(self, headers, wait_time):
        # Build mock response
        response = Response()
        response.status_code = 200
        response.headers = {'X-Throttling-Control': headers}
        
        # Get appropriate wait time from funciton
        test_wait_time = regulate_api_requests(response)

        # Ensure it matches what is expected
        assert test_wait_time == wait_time
