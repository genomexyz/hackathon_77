import numpy as np

#ftp://rainmap:Niskur+1404@hokusai.eorc.jaxa.jp/realtime

import ftplib
import os

# FTP server details
ftp_host = "hokusai.eorc.jaxa.jp"
ftp_user = "rainmap"
ftp_pass = "Niskur+1404"
ftp_directory = "/realtime_ver/v6/daily/00Z-23Z/202408"

# Local directory to save the downloaded files
local_directory = "data"

# Function to download a file from the FTP server
def download_file(ftp, filename):
    local_filepath = os.path.join(local_directory, filename)
    with open(local_filepath, 'wb') as local_file:
        ftp.retrbinary(f"RETR {filename}", local_file.write)
    print(f"Downloaded: {filename}")

# Main script
def main():
    # Ensure the local directory exists
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)

    # Connect to the FTP server
    ftp = ftplib.FTP(ftp_host)
    ftp.login(ftp_user, ftp_pass)
    ftp.cwd(ftp_directory)

    # List files in the directory
    files = ftp.nlst()

    # Download each file
    for filename in files:
        download_file(ftp, filename)

    # Close the FTP connection
    ftp.quit()

if __name__ == "__main__":
    main()
