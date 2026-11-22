from app.gmail import Gmail


if __name__ == "__main__":
    gmail = Gmail()
    print("Gmail connected. Inbox accessible:", bool(gmail.list_new_message_ids(max_results=1)))
