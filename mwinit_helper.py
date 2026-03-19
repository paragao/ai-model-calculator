#!/usr/bin/python3
import json
import subprocess
from pathlib import Path
from configparser import ConfigParser

ACCOUNT_EMAIL = "compute-sa-team@amazon.com"


def copy_credentials_to_aws(profile_name: str, output: str) -> dict:
    """
    Copy AWS credentials from JSON string to AWS credentials file.

    Args:
        profile_name: The name of the AWS profile to create/update
        output: JSON string with AWS credentials

    Returns:
        dict: Status information including success status and message

    Example:
        output = '{"Version": 1, "AccessKeyId": "ASIA...", "SecretAccessKey": "...",
                   "SessionToken": "...", "Expiration": "2025-12-02T07:18:11Z"}'
        result = copy_credentials_to_aws('my-profile', output)
    """
    try:
        # Parse the JSON from the output
        if not output or len(output) == 0:
            return {"success": False, "message": "output is empty"}

        # Parse the JSON string
        creds_data = json.loads(output)

        # Extract credentials
        access_key = creds_data.get("AccessKeyId")
        secret_key = creds_data.get("SecretAccessKey")
        session_token = creds_data.get("SessionToken")
        expiration = creds_data.get("Expiration")

        if not all([access_key, secret_key, session_token]):
            return {
                "success": False,
                "message": "Missing required credentials in output",
            }

        # Define credentials file path
        credentials_path = Path.home() / ".aws" / "credentials"

        # Create .aws directory if it doesn't exist
        credentials_path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing credentials file or create new ConfigParser
        config = ConfigParser()
        if credentials_path.exists():
            config.read(credentials_path)

        # Add or update the profile section
        if not config.has_section(profile_name):
            config.add_section(profile_name)

        config.set(profile_name, "aws_access_key_id", access_key)
        config.set(profile_name, "aws_secret_access_key", secret_key)
        config.set(profile_name, "aws_session_token", session_token)

        # Write the updated credentials back to file
        with open(credentials_path, "w") as f:
            config.write(f)

        return {
            "success": True,
            "message": f'Successfully copied credentials to profile "{profile_name}"',
            "profile_name": profile_name,
            "credentials_path": str(credentials_path),
            "expiration": expiration,
            "access_key": access_key[:8]
            + "..."
            + access_key[-4:],  # Masked for security
        }

    except json.JSONDecodeError as e:
        return {"success": False, "message": f"Failed to parse JSON: {str(e)}"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


output_list = subprocess.run(
    ["isengardcli", "credentials", ACCOUNT_EMAIL, "--role", "Admin", "--json"],
    capture_output=True,
    text=True,
)

result = copy_credentials_to_aws("default", output_list.stdout)

print(f"Success: {result['success']}")
print(f"Message: {result['message']}")
if result["success"]:
    print(f"Profile: {result['profile_name']}")
    print(f"Credentials Path: {result['credentials_path']}")
    print(f"Expiration: {result['expiration']}")
    print(f"Access Key (masked): {result['access_key']}")
