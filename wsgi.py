from app import app

# Vercel and other WSGI hosts expect a callable named `application`.
application = app

if __name__ == "__main__":
    application.run()
