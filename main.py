import streamlit as st

st.set_page_config(layout="wide")

panel_fit = st.Page(
            page  = "views/panel_fit.py",
            title = "Panel Assessment",
            icon  = ":material/roofing:"
        )

roof_dsc = st.Page(
            page  = "views/roof_dsc.py",
            title = "Locate Roof",
            icon  = ":material/home_pin:"
        )

about = st.Page(
            page  = "views/about.py",
            title = "Read me",
            icon  = ":material/local_library:"
        )

pages = st.navigation({ "Tools" : [roof_dsc, panel_fit],
                        "About" : [about]
                       })

pages.run()