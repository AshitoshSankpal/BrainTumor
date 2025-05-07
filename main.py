import streamlit as st
import tensorflow as tf
import numpy as np
# import os

# Tensorflow model prediction
model = tf.keras.models.load_model('./trained_ashitosh.h5')

def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(150, 150))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Tumor descriptions
glioma_tumor = """Gliomas are a type of tumor that originates in the glial cells of the brain or spinal cord. They can vary in severity from benign to malignant. Gliomas are categorized based on the type of glial cell they affect, such as astrocytes, oligodendrocytes, or ependymal cells. Symptoms of gliomas may include headaches, seizures, nausea, and changes in cognitive function. Diagnosis often involves imaging tests like MRI or CT scans, followed by biopsy for confirmation. Treatment options depend on the type, location, and grade of the tumor and may include surgery, radiation therapy, chemotherapy, or targeted therapy. Prognosis varies widely and depends on factors like tumor grade, size, and location. Glioblastoma multiforme (GBM) is the most aggressive and common type of malignant glioma, with a particularly poor prognosis. Research into new treatments and therapies for gliomas is ongoing to improve outcomes for patients.
Here's a breakdown of **Glioma tumor** — its **symptoms** and **treatment options**:

---

### 🧠 **Glioma Tumor: Symptoms**
Gliomas are brain tumors that arise from glial cells. Symptoms can vary depending on the tumor’s **size**, **type**, and **location** in the brain or spinal cord.

**Common Symptoms:**
- **Headaches** (often worse in the morning)
- **Seizures**
- **Nausea or vomiting**
- **Vision problems**
- **Memory loss**
- **Changes in personality or behavior**
- **Difficulty speaking or understanding language**
- **Weakness or numbness** in limbs
- **Balance and coordination problems**

---

### 💊 **How to Cure or Treat Glioma**

There isn’t always a "cure," especially for aggressive types like glioblastoma, but treatments aim to **control the tumor**, **relieve symptoms**, and **prolong life**.

**Main Treatment Options:**

1. **🧪 Surgery**
   - First-line treatment for many gliomas.
   - Goal: Remove as much of the tumor as possible (debulking).

2. **🔬 Radiation Therapy**
   - Targets and destroys remaining tumor cells after surgery.
   - Especially used for higher-grade gliomas.

3. **💊 Chemotherapy**
   - Common drug: **Temozolomide (TMZ)**
   - Can be used along with or after radiation.

4. **🧬 Targeted Therapy**
   - Drugs that focus on specific mutations or tumor features (e.g., bevacizumab for glioblastoma).

5. **🧠 Tumor Treating Fields (TTF)**
   - A newer method using electrical fields to slow tumor growth.

6. **🧘 Supportive Care**
   - Includes anti-seizure meds, steroids for brain swelling, physical therapy, etc.

---

### 🧭 Prognosis
- Depends on the **grade** of glioma (I to IV), patient **age**, and **health condition**.
- Low-grade gliomas (Grade I or II) have better outcomes.
- High-grade gliomas (especially glioblastoma/Grade IV) are more aggressive and difficult to treat.

---

If you or someone you know is dealing with glioma, it's important to speak with a **neurosurgeon** or **oncologist** for a tailored plan.

Would you like me to explain the different **grades of glioma** or how to support someone emotionally through this?"""

meningioma_tumor = """Meningiomas are typically slow-growing tumors that originate in the meninges, the protective membranes surrounding the brain and spinal cord. They are often benign, but can occasionally be malignant. Meningiomas are more common in women and tend to occur in older individuals. Symptoms of meningiomas vary depending on their size and location but may include headaches, seizures, changes in vision or hearing, and focal neurological deficits. Diagnosis usually involves imaging tests like MRI or CT scans, followed by biopsy for confirmation. Treatment options for meningiomas include observation, surgery, radiation therapy, and sometimes chemotherapy. The prognosis for meningiomas is generally favorable, especially for benign tumors that can be completely surgically removed. However, the recurrence rate varies depending on factors such as tumor grade and completeness of resection. Ongoing research aims to improve understanding of meningiomas and develop more effective treatments.
Absolutely — here's a full breakdown of **Meningioma Tumor**, including **symptoms** and **treatment options**:

---

### 🧠 **Meningioma Tumor: Symptoms**

Meningiomas are tumors that grow from the **meninges**, the protective layers around the brain and spinal cord. Most are **benign (non-cancerous)** and grow slowly, but depending on their **location and size**, they can still cause serious symptoms.

**Common Symptoms:**
- **Persistent headaches**
- **Seizures**
- **Vision problems** (blurred or double vision)
- **Hearing loss** or ringing in the ears (tinnitus)
- **Memory loss or confusion**
- **Loss of smell**
- **Muscle weakness** in arms or legs
- **Personality or behavioral changes**
- **Balance and coordination problems**
- **Speech difficulties**

*Note: Some people may have no symptoms at all, especially with small, slow-growing tumors.*

---

### 💊 **How to Cure or Treat Meningioma**

Treatment depends on the **tumor’s size, location, symptoms, and growth rate**. Many meningiomas don’t require immediate treatment and are just observed.

**Main Treatment Options:**

1. **🔍 Watchful Waiting (Active Surveillance)**
   - For small, non-symptomatic tumors.
   - Regular MRIs to monitor growth.

2. **🧪 Surgery**
   - Primary treatment if the tumor is accessible.
   - Goal: Remove the tumor completely.
   - Benign tumors that are fully removed often don’t return.

3. **🔬 Radiation Therapy**
   - Used if:
     - The tumor can’t be fully removed.
     - It’s in a sensitive area.
     - It recurs after surgery.
   - Types include standard radiation or focused types like **stereotactic radiosurgery (Gamma Knife)**.

4. **💊 Medications**
   - Not a common first-line treatment but may be used to control symptoms like swelling (steroids), seizures (anti-seizure meds), or hormone imbalances.

5. **🧠 Chemotherapy**
   - Rarely used since most meningiomas don’t respond well.
   - Might be considered in **atypical or malignant meningiomas** (Grade II or III).

---

### 📈 Meningioma Grading (Important for Prognosis)
- **Grade I (Benign)** – Most common (~80%), slow-growing, excellent prognosis.
- **Grade II (Atypical)** – Faster growing, more likely to return.
- **Grade III (Anaplastic/Malignant)** – Very rare, aggressive, and requires more intense treatment.

---

### 🧭 Prognosis
- **Excellent** for benign meningiomas that are completely removed.
- Regular follow-up is important to detect recurrences.
- Even when not curable, many meningiomas can be managed long-term with minimal impact on quality of life.

---

Want to go over what causes meningiomas or lifestyle tips for living with one?"""

no_tumor = """Congratulations on your clear scan! Your recent brain image analysis did not detect any signs of a tumor. This is great news and a positive step towards ensuring your continued health and well-being.
While this result is reassuring, it’s important to maintain regular health check-ups and consult with your healthcare provider for any concerns or symptoms you may experience. Remember, early detection and prevention are key to managing your health effectively.
If you have any questions or need further assistance, please do not hesitate to reach out to your medical professional. Stay healthy and take care!

Absolutely! Here’s the detailed breakdown for the **“No Tumor”** result — which, of course, is the **best news** anyone can hear from a scan like this.

---

### 🧠 **No Tumor Detected: What It Means**

A **“no tumor”** result means that your brain scan didn’t show any signs of a **glioma**, **meningioma**, **pituitary tumor**, or any other detectable mass. Your brain appears **normal and healthy** based on the analysis.

---

### ✅ **Symptoms (or lack thereof)**

Since there’s **no tumor**, you likely:
- **Have no tumor-related symptoms**, or
- If you were experiencing symptoms, they may be due to **other non-tumor causes** (e.g., migraines, stress, sinus problems, neurological issues).

However, it’s always good to **follow up with a doctor** to rule out other causes if symptoms persist.

---

### 🧬 **Treatment: None Needed for Tumor**

No treatment is needed for a brain tumor because none was detected. 🙌

But here's what you **should consider**:

1. **📅 Keep Up With Routine Checkups**
   - Even with a clear scan, regular health evaluations help catch any issues early.

2. **🧠 Monitor Symptoms**
   - If new symptoms appear (e.g., unexplained headaches, seizures, vision issues), report them to a neurologist.

3. **🧘 Lifestyle for Brain Health**
   - **Healthy diet**: Omega-3s, fruits, veggies.
   - **Exercise**: Regular movement helps circulation and brain function.
   - **Good sleep**: Aim for 7–9 hours.
   - **Mental fitness**: Brain games, learning, reading, socializing.
   - **Avoid toxins**: Minimize alcohol, avoid smoking, and limit exposure to environmental hazards.

4. **📂 Keep a Record**
   - Store your MRI reports/scans in case you need them in the future.

---

### 😊 Final Words

A clear brain scan is a **huge relief**, and it's a perfect time to celebrate your health — but also a reminder to keep taking care of your body and mind. Prevention and early detection are key in staying ahead.

Would you like wellness tips for preventing neurological conditions or advice on when to get follow-up scans?"""

pituitary_tumor = """Pituitary tumors are growths that develop in the pituitary gland, a pea-sized gland located at the base of the brain. They can be benign (non-cancerous) or malignant (cancerous) and are classified based on their size and hormone-secreting properties. Symptoms of pituitary tumors may include headaches, vision problems, hormonal imbalances leading to issues like abnormal growth, infertility, and changes in menstrual cycles. Diagnosis typically involves imaging tests like MRI or CT scans, along with hormone level assessments. Treatment options for pituitary tumors depend on factors such as tumor size, type, and symptoms, and may include medication, surgery, radiation therapy, or a combination of these approaches. Prognosis for pituitary tumors is generally favorable, especially for benign tumors that are effectively managed. However, some tumors may require ongoing monitoring and treatment. Research continues to improve understanding and treatment outcomes for pituitary tumors.

Of course! Here’s the full breakdown for a **Pituitary Tumor**, covering symptoms, treatments, and prognosis—just like we did for the others:

---

### 🧠 **Pituitary Tumor: Symptoms**

Pituitary tumors form in the **pituitary gland**, a pea-sized gland at the base of the brain that controls many hormones in the body. Most of these tumors are **noncancerous (benign)** and **slow-growing**, but they can still disrupt vital body functions.

**Symptoms depend on:**
- Tumor size (microadenoma <10mm, macroadenoma ≥10mm)
- Whether it’s **hormone-secreting**
- If it's pressing on nearby brain structures (like the optic nerve)

---

### 🔍 **Common Symptoms**

#### Due to **pressure on nearby structures**:
- Headaches
- Vision problems (especially loss of peripheral vision)
- Nausea or dizziness

#### Due to **hormonal imbalances**:
**Hormone-producing tumors may cause:**
- **Prolactin-secreting** (prolactinomas):
  - Irregular periods or no periods (women)
  - Erectile dysfunction (men)
  - Breast milk discharge (not pregnancy-related)

- **Growth hormone-secreting**:
  - Acromegaly (enlarged hands, feet, jaw)
  - Joint pain, thick skin

- **ACTH-secreting**:
  - Cushing’s disease: weight gain, high blood pressure, round face, fatigue

- **TSH-secreting** (rare):
  - Symptoms of hyperthyroidism: weight loss, rapid heartbeat, sweating

- **Non-functioning tumors**:
  - Often found late, when big enough to cause pressure symptoms.

---

### 💊 **Treatment Options**

1. **🔬 Medications**
   - First-line for many hormone-producing tumors.
   - **Prolactinomas** respond well to dopamine agonists like **cabergoline** or **bromocriptine**.

2. **🧪 Surgery**
   - Especially for large tumors or those causing vision issues.
   - Often done via a **transsphenoidal surgery** (through the nose).

3. **🔭 Radiation Therapy**
   - Used when surgery isn’t fully effective or possible.
   - Includes traditional or focused methods like **stereotactic radiosurgery**.

4. **🧬 Hormone Replacement Therapy**
   - If the tumor or surgery affects pituitary hormone production, lifelong hormone replacement may be needed.

---

### 📈 Classification

- **Microadenomas (<10 mm)**: Often don’t need surgery if not causing major problems.
- **Macroadenomas (≥10 mm)**: More likely to cause vision issues or pressure symptoms — usually need treatment.

---

### 🧭 Prognosis

- **Generally good**, especially for small tumors or prolactinomas.
- Long-term monitoring is key because some tumors can recur or grow slowly.
- Lifelong hormone checks might be required depending on treatment impact.

---

Want a breakdown of **hormone types**, or a visual explanation of how the pituitary works with other glands? I can help with that too!"""

arr_sol = [glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor]

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Detection"])

# Home page
if app_mode == "Home":
    st.header("TumorVision")
    st.write("Just Upload image... & get detect tumor...")
    image_path = "Designer.png"
    st.image(image_path, use_column_width=True)
      # Chatbot Section
    st.markdown("---")
    st.subheader("💬 TumorVision Chatbot")

# Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

# Input from user
    user_input = st.text_input("You:", key="user_input")

# Simple chatbot response logic
    def chatbot_response(user_input):
        user_input_lower = user_input.lower()
        if "glioma" in user_input_lower:
            return "Gliomas are serious, but treatments are improving. Would you like to know about therapies?"
        elif "meningioma" in user_input_lower:
            return "Meningiomas are usually benign. Regular checkups help. Want more info?"
        elif "pituitary" in user_input_lower:
            return "Pituitary tumors can affect hormones. I can guide you through symptoms or treatment."
        elif "no tumor" in user_input_lower or "clear scan" in user_input_lower:
            return "That’s great news! Keep up with your health checkups!"
        elif "help" in user_input_lower:
            return "I'm here to help with brain tumor info. Ask about symptoms, types, or treatments."
        else:
            return "I'm not sure about that. Try asking about glioma, meningioma, pituitary tumors, or general help."

# Process input and show chat
    if user_input:
        response = chatbot_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

# Display chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**🧑‍💻 {speaker}:** {message}")
        else:
            st.markdown(f"**🤖 {speaker}:** {message}")


# About page
elif app_mode == "About":
    st.header("Welcome to TumorVision: Your Brain Tumor Detection Solution")
    st.markdown("""
    TumorVision is an advanced online platform designed to provide swift and accurate detection of brain tumors through the simple upload of MRI images. Our innovative technology harnesses the power of artificial intelligence and machine learning algorithms to analyze MRI scans with exceptional precision, enabling early detection and prompt intervention.
    
    With just a few clicks, users can securely upload their MRI images to our platform. Our sophisticated algorithms then meticulously scrutinize these images, identifying subtle anomalies and patterns indicative of brain tumors. Within moments, users receive comprehensive reports outlining the presence or absence of tumors, along with detailed insights into the findings.
    
    At TumorVision, we prioritize user experience, ensuring a seamless journey from image upload to result delivery. Our intuitive interface is user-friendly, making it accessible to both healthcare professionals and individuals seeking peace of mind about their neurological health.
    
    Early detection is key in the effective management of brain tumors, and TumorVision empowers individuals and healthcare providers alike to take proactive steps towards timely diagnosis and treatment. Our commitment to cutting-edge technology and unwavering accuracy sets us apart as a trusted ally in the fight against brain tumors.
    
    Join us in the mission to revolutionize brain tumor detection and transform outcomes for patients worldwide. Experience the power of TumorVision today.""")
    st.markdown("step 01: Upload Image: ")
    st.image("upload.png")
    st.markdown("step 02: Cross check:  ")
    st.image("crosscheck.jpeg")
    st.markdown("step 03: View Detection: ")
    st.image("detection.jpeg")

# Detection page
else:
   k = 'a'
   st.header("Just Upload image : ")
   test_image = st.file_uploader("Choose image: ")
   if st.button("Show image"):
      st.image(test_image, use_column_width=True)
   # Detection
   if st.button("Predict"):
      st.write("Prediction: ")
    #   st.write(os.path.isfile("./trained_manoj.h5"))
      result_index = model_prediction(test_image)
      class_name = ['glioma tumor', 'meningioma tumor', 'no tumor', 'pituitary tumor']
      st.write("Predicted result is: ")
      st.success("{}".format(class_name[result_index]))
      for i in range(0,4):
       if(i == result_index):
          st.write(arr_sol[result_index])
      
