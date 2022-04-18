import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():
    st.title("Mushroom Classification Using Machine Learning")
    st.sidebar.title("Try out with different Machine Learning Algorithms to get the best results")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")
    # One good thing about streamlit : It only updates the changes and # not
    # creates the whole web page again and again :: Thus, very fast + intelligent

    # Step 1 : Prepare the dataset
    # Decorator : this will avoid re-loading of the data each time we run the app and thus saves time
    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('./mushrooms.csv')
        le = LabelEncoder()
        data = data.apply(le.fit_transform)
        return data


    # Step 2 : Train and Test split
    # Use of cache : If the ratio remains same, we would not process it again and thus it will save time
    @st.cache(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test


    # Step 3 : Visualising different ml based curves
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Curve')
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()



    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible' , 'poisonous']

    # Step 4 : Allowing the user to choose the classifier
    st.sidebar.subheader("Choose Classifier")
    # Drop-down menu
    classifier = st.sidebar.selectbox("Classifier",("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest Classifier"))

    # Step 5 : Applying ML algo according to the user

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        # Inputting the Hyperparameter within a range
        C = st.sidebar.number_input("C (Penalty factor)", 0.01, 100.0, step= 0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf" , "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale","auto"), key="gamma")

        # Providing Plotting options (Multiple Plot Selection)
        metrics = st.sidebar.multiselect("What metrics to plot ?" , ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        # The user will click multiple options
        # And we don't want the web page to reload again!!
        # Thus, this functionality is added
        if st.sidebar.button("Classify", key='classify'):
            st.subheader('Support Vector Machine (SVM) Results')

            # Training the model
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            # Displaying the Results
            st.write("Accuracy : ", accuracy.round(2))
            st.write("Precision : ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall : ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)



    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        # Inputting the Hyperparameter within a range
        C = st.sidebar.number_input("C (Penalty factor)", 0.01, 100.0, step= 0.01, key='C_LR')
        max_iter = st.sidebar.slider("Max epochs" , 100, 500 , key="max_iter")

        # Providing Plotting options (Multiple Plot Selection)
        metrics = st.sidebar.multiselect("What metrics to plot ?" , ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        # The user will click multiple options
        # And we don't want the web page to reload again!!
        # Thus, this functionality is added
        if st.sidebar.button("Classify", key='classify'):
            st.subheader('Logistic Regression Results')

            # Training the model
            model = LogisticRegression(C=C,max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            # Displaying the Results
            st.write("Accuracy : ", accuracy.round(2))
            st.write("Precision : ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall : ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)


    if classifier == 'Random Forest Classifier':
        st.sidebar.subheader("Model Hyperparameters")
        # Inputting the Hyperparameter within a range
        n_estimators = st.sidebar.number_input("The number of tress in the forest", 100, 500, step=10, key="n_estimators")
        max_depth = st.sidebar.number_input("Max depth of each tree", 1, 20, step=1, key="max_depth")
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", (True, False) , key="bootstrap")
        criterion = st.sidebar.radio("Information Gain criterion",('entropy','gini') , key="criterion")

        # Providing Plotting options (Multiple Plot Selection)
        metrics = st.sidebar.multiselect("What metrics to plot ?" , ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        # The user will click multiple options
        # And we don't want the web page to reload again!!
        # Thus, this functionality is added
        if st.sidebar.button("Classify", key='classify'):
            st.subheader('Random Forest Classifier Results')

            # Training the model
            model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, bootstrap=bootstrap, max_depth=max_depth)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            # Displaying the Results
            st.write("Accuracy : ", accuracy.round(2))
            st.write("Precision : ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall : ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)


    # To view the dataset; we will add a checkbox
    if  st.sidebar.checkbox("Show raw data" , False):
        st.subheader("Mushroom Data Set (used for Classification)")
        st.write(df)

if __name__ == '__main__':
    
    main()
