import streamlit as st
import tempfile

from login import login
from detector import detect_image
from face_recognition import compare_faces

st.set_page_config(
page_title="AI Forensic",
layout="wide"
)

with open("style.css") as f:
 st.markdown(
 f"<style>{f.read()}</style>",
 unsafe_allow_html=True
 )

st.title("AI Deepfake & Face Forensics")

if "logged" not in st.session_state:
 st.session_state.logged=False

if not st.session_state.logged:

 user=st.text_input("Username")
 password=st.text_input("Password",type="password")

 if st.button("Login"):

   if login(user,password):
      st.session_state.logged=True
      st.rerun()

   else:
      st.error("Wrong Credentials")

else:

 tab1,tab2=st.tabs([
 "Detection",
 "Face Match"
 ])

 with tab1:

   img=st.file_uploader("Upload Image")

   if img:

      status,score,image=detect_image(img)

      st.image(image)

      st.write(status)

      st.metric(
      "Score",
      str(score)+"%"
      )

 with tab2:

   img1=st.file_uploader("Image1")
   img2=st.file_uploader("Image2")

   if img1 and img2:

      def save(file):
        t=tempfile.NamedTemporaryFile(delete=False,suffix=".jpg")
        t.write(file.read())
        t.close()
        return t.name

      p1=save(img1)
      p2=save(img2)

      result=compare_faces(p1,p2)

      st.metric(
      "Similarity",
      str(result["similarity"])+"%"
      )

      st.success(result["status"])
