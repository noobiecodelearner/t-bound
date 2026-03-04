"""
[api/db/database). — SQLAlchemy database setup.

DAYANCH — implement this file.

What to do:
    1. Create a SQLAlchemy engine pointing to SQLite at 'tbound.db'
    2. Create a SessionLocal factory using sessionmaker
    3. Create a Base class using declarative_base()
    4. Write a get_db() dependency function for FastAPI dependency injection
       that yields a session and closes it after the request

Example structure:
    DATABASE_URL = "sqlite:///./tbound.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

Dependencies:
    pip install sqlalchemy
"""
