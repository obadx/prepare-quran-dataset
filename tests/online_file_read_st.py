import streamlit as st
from pathlib import Path
import time
import multiprocessing
import concurrent
import os
import signal
import sys
from dataclasses import dataclass


@dataclass
class Number:
    x: int


class ProcessTermination(Exception):
    ...


def hamo():
    print('hamo')


@st.dialog("Stop Process?")
def stop_process_popup(pid: int):
    col1, col2 = st.columns(2)
    placeholder = st.empty()

    with placeholder.container():
        with col1:
            if st.button("Yes", use_container_width=True,):
                try:
                    kill_process(pid)
                    placeholder.success('Download is canceled')
                except Exception as e:
                    placeholder.error(f'Error While canceling Download: {e}')
                    raise e
                st.rerun()
        with col2:
            if st.button("No", use_container_width=True):
                placeholder.info("Aborting ...")
                time.sleep(1)
                st.rerun()


def kill_process(pid):
    try:
        os.kill(pid, signal.SIGTERM)  # or signal.SIGKILL for force killing
        print(f"Process {pid} has been terminated.")
    except ProcessLookupError:
        print(f"No process found with PID {pid}.")
    except PermissionError:
        print(f"No permission to terminate process {pid}.")


@st.cache_data(ttl=1)
def read_file(file) -> str:
    with open(file, 'r') as f:
        out = f.read()
    return out


def task(idx: int):
    while True:
        print(f'{idx} -> running')
        time.sleep(5)


def multiple_tasks(num: Number):
    print('Number: ', num.x)
    # create a handler for watching for SIGTERM
    # signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        hamo()
        num = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=num) as executor:
            executor.map(task, list(range(num)))
        # from pypdl import Pypdl
        # dl = Pypdl()
        # out = dl.start(
        #     'https://download.quran.islamway.net/quran3/965/212/128/002.mp3', segments=20)
    except ProcessTermination:
        print('catch exception')


def handle_sigterm(signum, frame):
    raise ProcessTermination("SIGTERM received")


def worker_task():

    # Set the start method to 'spawn' to create a fully detached process.
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=multiple_tasks, args=(Number(x=10),))
    # Detach the process to prevent it from being terminated with the main program
    p.daemon = True
    p.start()
    print(f'Process PID: {p.pid}')
    st.session_state.process = p
    st.session_state.process_pid = p.pid


if __name__ == '__main__':

    if 'pressed' not in st.session_state:
        st.session_state.pressed = False

    if 'start_process' not in st.session_state:
        st.session_state.start_process = False

    if 'stop_process' not in st.session_state:
        st.session_state.stop_process = False

    file = Path('hamo.txt')
    if file.is_file():
        st.write(read_file(file))

        if st.button('Start') and 'process' not in st.session_state:
            st.session_state.start_process = True

        if st.button('Stop'):
            st.session_state.stop_process = True

        if st.session_state.start_process:
            print('Start Pressed')
            worker_task()
            # st.session_state['process'] = True  # TODO: for testing only
            st.session_state.start_process = False

        if st.session_state.stop_process and 'process' in st.session_state:
            print('Stop Pressed')
            # st.session_state.process.terminate()
            # del st.session_state['process']
            stop_process_popup(st.session_state.process_pid)
            st.session_state.stop_process = False
        else:
            time.sleep(2)
            st.rerun()
