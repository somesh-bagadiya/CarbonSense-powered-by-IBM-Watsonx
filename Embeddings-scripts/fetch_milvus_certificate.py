import os
import ssl
import socket
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv(override=True)

# Read Milvus host and port from .env
host = os.getenv("MILVUS_GRPC_HOST")
port = int(os.getenv("MILVUS_GRPC_PORT", 30902))  # default port if missing

# Output certificate file
cert_file = "milvus-grpc.crt"

print(f"Fetching certificate from {host}:{port}...")

try:
    # Attempt to fetch server certificate
    cert = ssl.get_server_certificate((host, port))

    # Write the certificate to a file
    with open(cert_file, "w") as f:
        f.write(cert)

    print(f"✅ Certificate successfully saved to: {cert_file}")

except socket.gaierror as e:
    print(f"❌ DNS resolution failed for host: {host}")
    print(f"Error: {e}")

except Exception as e:
    print(f"❌ Failed to fetch certificate from {host}:{port}")
    print(f"Error: {e}")
