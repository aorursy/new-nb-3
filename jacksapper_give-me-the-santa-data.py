#Get data from behind a firewall that blocks Google CDN



from subprocess import check_output

import os

print(check_output(["ls", "../input"]).decode("utf8"))






# Any results you write to the current directory are saved as output.