from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, InterfaceError, ProgrammingError

def check_db_connection(engine):
    #Checks if a database connection can be established.
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        conn_ = True
        return conn_
    except (OperationalError, InterfaceError, ProgrammingError) as e:
        conn_ = False
        return conn_
    except Exception as e:
        conn_ = False
        return conn_
    
def main(username, password, host, port, db_name):
    #Constructs the database connection string and checks the connection.
    db_info = f"mysql+pymysql://{username}:{password}@{host}:{port}/{db_name}"
    engine = create_engine(db_info)
    conn_ = check_db_connection(engine)
    return db_info, engine, conn_

if __name__ == "__main__":
    main()