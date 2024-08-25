import streamlit as st

from menu import menu_with_redirect


def main():
    menu_with_redirect()
    st.header('Recitation Database Manager')


if __name__ == '__main__':
    main()
