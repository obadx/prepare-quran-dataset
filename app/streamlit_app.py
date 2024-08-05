import streamlit as st
from prepare_quran_dataset.construct.database import ReciterPool, MoshafPool
from prepare_quran_dataset.construct.data_classes import Reciter, Moshaf
from pydantic import BaseModel, Field
from typing import Type, Any, Literal

# Initialize pools
reciter_pool = ReciterPool()
moshaf_pool = MoshafPool(reciter_pool)


def main():
    st.title("Quran Dataset Management")

    menu = ["Reciter Pool", "Moshaf Pool"]
    choice = st.sidebar.selectbox("Select Pool", menu)

    if choice == "Reciter Pool":
        pool_interface(Reciter, reciter_pool, "Reciter")
    elif choice == "Moshaf Pool":
        pool_interface(Moshaf, moshaf_pool, "Moshaf")


def pool_interface(model: Type[BaseModel], pool, pool_name: str):
    st.header(f"{pool_name} Pool")

    operation = st.radio("Choose operation", ["Insert", "View", "Update"])

    if operation == "Insert":
        insert_item(model, pool)
    elif operation == "View":
        view_items(pool)
    elif operation == "Update":
        update_item(model, pool)


def insert_item(model: Type[BaseModel], pool):
    st.subheader(f"Insert New {model.__name__}")

    form_data = {}
    for field_name, field in model.__fields__.items():
        if field_name != 'id':  # Exclude 'id' field as it's auto-generated
            field_input = create_input_for_field(field, for_insertion=True)
            if field_input is not None:
                form_data[field_name] = field_input

    if st.button(f"Insert {model.__name__}"):
        try:
            new_item = model(**form_data)
            pool.insert(new_item)
            st.success(f"{model.__name__} inserted successfully!")
        except Exception as e:
            st.error(f"Error inserting {model.__name__}: {str(e)}")


def view_items(pool):
    st.subheader(f"View {pool.item_type.__name__}s")

    for item in pool:
        for field_name, field in item.__class__.__fields__.items():
            value = getattr(item, field_name)
            label = field_name
            arabic_name = get_arabic_name(field)
            label = f"{arabic_name} ({field_name})"
            st.write(f"{label}: {value}")
        st.write("---")


def update_item(model: Type[BaseModel], pool):
    st.subheader(f"Update {model.__name__}")

    item_id = st.text_input(f"Enter {model.__name__} ID to update")

    try:
        item = pool[item_id]
        form_data = {}
        for field_name, field in model.__fields__.items():
            if field_name != 'id':
                current_value = getattr(item, field_name)
                form_data[field_name] = create_input_for_field(
                    field, current_value)

        if st.button(f"Update {model.__name__}"):
            try:
                updated_item = model(id=item_id, **form_data)
                pool.update(updated_item)
                st.success(f"{model.__name__} updated successfully!")
            except Exception as e:
                st.error(f"Error updating {model.__name__}: {str(e)}")
    except KeyError:
        st.error(f"{model.__name__} not found")


def create_input_for_field(field, default_value=None, for_insertion=False):
    # Skip fields with default values or default_factory during insertion
    if for_insertion and (field.default is not None or field.default_factory is not None):
        return None

    # Extract Arabic name from field description if available
    label = field.name
    arabic_name = get_arabic_name(field)
    label = f"{arabic_name} ({field.name})"

    if hasattr(field.type_, "__origin__") and field.type_.__origin__ is Literal:
        choices = list(field.type_.__args__)
        return st.selectbox(label, choices, index=choices.index(default_value) if default_value in choices else 0)
    elif field.type_ == str:
        return st.text_input(label, value=default_value or "")
    elif field.type_ == int:
        return st.number_input(label, value=default_value or 0, step=1)
    elif field.type_ == float:
        return st.number_input(label, value=default_value or 0.0, step=0.1)
    elif field.type_ == bool:
        return st.checkbox(label, value=default_value or False)
    elif field.type_ == list:
        if default_value is None:
            default_value = []
        return st.text_area(f"{label} (one per line)", value="\n".join(map(str, default_value))).split("\n")
    else:
        st.warning(f"Unsupported field type for {label}: {field.type_}")
        return None


def get_arabic_name(field: Field) -> str:
    """get the Arabic name out of the field description
    """
    if field.field_info.description and 'ArabicName' in field.field_info.description:
        arabic_name = field.field_info.description.split(
            'ArabicName(')[1].split(')')[0]
        return arabic_name
    return ''


if __name__ == "__main__":
    main()
