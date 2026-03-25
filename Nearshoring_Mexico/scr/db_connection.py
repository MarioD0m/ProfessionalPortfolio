from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, InterfaceError, ProgrammingError

def check_db_connection(engine):
    #Checks if a database connection can be established.
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("\n Database connection successful!")
        conn_ = True
        return conn_
    except (OperationalError, InterfaceError, ProgrammingError) as e:
        print(f"\n Database connection failed. Error: {e}")
        conn_ = False
        return conn_
    except Exception as e:
        print(f"\n An unexpected error occurred. Error: {e}")
        conn_ = False
        return conn_
    
def main():
    print('\nINITIATING DATABASE CONNECTION CHECK ...')
    print('\n -> Input your credentials for database connection:')
    username = input(" --> Username: ")
    password = input(" --> Password: ")
    host = input(" --> Host: ")
    port = input(" --> Port: ")
    db_name = input(" --> Database Name: ")
    db_table = input(" --> Table Name: ")
    db_info = f"mysql+pymysql://{username}:{password}@{host}:{port}/{db_name}"
    engine = create_engine(db_info)
    conn_ = check_db_connection(engine)
    return db_info, db_table, engine, conn_

if __name__ == "__main__":
    main()
