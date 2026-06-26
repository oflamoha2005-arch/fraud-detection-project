import os
import smtplib
import traceback
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

SMTP_SERVER = os.getenv("ALERT_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("ALERT_SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("ALERT_SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("ALERT_SMTP_PASSWORD")
RECIPIENT = os.getenv("ALERT_EMAIL")

msg = f"Subject: SMTP Test\n\nThis is a test from the fraud app."

print("SMTP debug:")
print("  server:", SMTP_SERVER)
print("  port:", SMTP_PORT)
print("  username:", SMTP_USERNAME)
print("  recipient:", RECIPIENT)
print("  password present:", bool(SMTP_PASSWORD))

try:
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as smtp:
        smtp.starttls()
        smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
        smtp.sendmail(SMTP_USERNAME, RECIPIENT, msg)
    print('SMTP test message sent')
except Exception as e:
    print('SMTP test failed:')
    print(str(e))
    traceback.print_exc()
    raise