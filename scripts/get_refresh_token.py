from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
# prompt="consent" makes Google return a refresh token every time.
creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")

print("OAuth authorization completed.")
print("Refresh token obtained:", bool(creds.refresh_token))