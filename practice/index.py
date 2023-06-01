import os

def run():
    from src.tasks.tasks import main as run_tasks
    RESOURCES_DIR = os.getcwd()

    print(RESOURCES_DIR)
    run_tasks()




if __name__ == '__main__':
    run()
else:
    print('index.py has not been run')
