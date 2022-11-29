export DISPLAY=$(ip route list default | awk '{print $3}'):0
