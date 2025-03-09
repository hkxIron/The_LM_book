import socket

def find_free_port():
    master = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    master.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    master.bind(('localhost', 0))
    _, port = master.getsockname()
    master.close()
    return port

free_port = find_free_port()
print(free_port)