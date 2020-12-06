import streamlit as st
import pandas as pd
import pickle

st.write("""
# Mini Project!
""")

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')



def get_input():
    #widgets
    v_AcademicYear = st.sidebar.radio('AcademicYear', ['2562','2563'])
    v_Sex = st.sidebar.radio('Sex',['Male','Female'])
    # v_FacultyID = st.sidebar.slider('FacultyID', , , )
    # v_DepartmentCode = st.sidebar.slider('DepartmentCode', , , )
    # v_EntryTypeID = st.sidebar.slider('EntryTypeID', , , )
    # v_EntryGroupID = st.sidebar.slider('EntryGroupID', , , )
    v_TCAS = st.sidebar.radio('TCAS',[ 1 , 2 , 3 , 4 , 5] )
    v_HomeRegion = st.sidebar.selectbox('HomeRegion',[ 'Bangkok' , 'Central' , 'East' , 'International' , 'North' , 'North East' , 'South' , 'West' ])
    # v_StudentTH = st.sidebar.slider('StudentTH ', , , )
    # v_ProvinceNameEng = st.sidebar.slider('ProvinceNameEng', , , )
    # v_NationName = st.sidebar.slider('NationName', , , )
    # v_ReligionName = st.sidebar.slider('ReligionName', , , )
    v_GPAX = st.sidebar.slider('GPAX',min_value=0.00, max_value=4.00)
    # v_GPA_Eng = st.sidebar.slider('GPA_Eng', , , )
    # v_GPA_Math = st.sidebar.slider('GPA_Math', , , )
    # v_GPA_Sci = st.sidebar.slider('GPA_Sci', , , )
    # v_GPA_Sco = st.sidebar.slider('GPA_Sco', , , )
    v_NumAnwser = st.sidebar.slider('NumAnwser',min_value=0, max_value=42)






    if v_AcademicYear == '2562': v_AcademicYear = '0'
    else : v_AcademicYear = '1'

    if v_Sex == 'Female': v_Sex = '0'
    else: v_Sex = '1'



    # if v_ProvinceNameEng == '': v_ProvinceNameEng = '0'
    # elif v_ProvinceNameEng == '': v_ProvinceNameEng = '1'

    
    


   

    #dictionary
    data = {'AcademicYear': v_AcademicYear,
            'Sex': v_Sex,
            # 'FacultyID': v_FacultyID,
            # 'DepartmentCode': v_DepartmentCode,
            # 'EntryTypeID': v_EntryTypeID,
            # 'EntryGroupID': v_EntryGroupID,
            'TCAS': v_TCAS,
            'HomeRegion': v_HomeRegion,
            # 'StudentTH': v_StudentTH,
            # 'ProvinceNameEng': v_ProvinceNameEng,
            # 'NationName': v_NationName,
            # 'ReligionName': v_ReligionName,
            'GPAX': v_GPAX,
            # 'GPA_Eng': v_GPA_Eng,
            # 'GPA_Math': v_GPA_Math,
            # 'GPA_Sci': v_GPA_Sci,
            # 'GPA_Sco': v_GPA_Sco,
            'NumAnwser':v_NumAnwser


            }

    #create data frame
    data_df = pd.DataFrame(data, index=[0])
    return data_df

df = get_input()
st.write(df)

data_sample = pd.read_csv('new_samples2.csv')
data_sample = data_sample.drop(columns=['Status'])
df = pd.concat([df, data_sample],axis=0)

cat_data = pd.get_dummies(df[['HomeRegion']])

#Combine all transformed features together
X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1] # Select only the first row (the user input data)
#Drop un-used feature
X_new = X_new.drop(columns=['HomeRegion'])


# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization2.pkl', 'rb'))
# #Apply the normalization model to new data
X_new = load_nor.transform(X_new)
# st.write(X_new)

# # -- Reads the saved classification model
load_knn = pickle.load(open('best_knn2.pkl', 'rb'))
# # Apply model for prediction
prediction = load_knn.predict(X_new)
st.write(prediction)