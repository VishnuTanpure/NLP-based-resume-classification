# NLP-based-resume-classification
NLP based Machine Learning project
![Resume Classification Screenshot](https://github.com/VishnuTanpure/NLP-based-resume-classification/blob/main/Screenshot%202024-09-20%20235522.png)

---
### Project Overview

The **Resume Classification** project automates the classification of resumes into four categories:
1. ReactJS
2. Peoplesoft
3. SQL
4. Workday

This project is designed to streamline resume sorting and quickly identify the relevant skillsets required for specific job roles. It uses a machine learning model to classify resumes based on their content. The classifier is implemented using a Support Vector Classifier (SVC) and the model is fine-tuned with GridSearchCV for optimal performance. The input formats supported are `.pdf` and `.docx`.

### Features

- **Text Processing**: Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction.
- **Customizable Classification**: A threshold slider allows users to control the confidence level of classification. If the probability of a resume belonging to one of the four categories is below the set threshold, the resume is classified as 'Other'.
- **Resume Sorting**: The project creates a directory in the system where subfolders are generated for each category (ReactJS, Peoplesoft, SQL, Workday, Other). Resumes are automatically sorted and placed into their respective folders.
- **CSV Download**: Users can download a CSV file that lists the file name and its assigned category, allowing easy tracking of the classification results.
- **Supports Multiple File Formats**: The model accepts resumes in `.pdf` and `.docx` formats.

### Model Workflow

1. **Input**: Upload a resume file in `.pdf` or `.docx` format.
2. **Text Processing**: The content of the resume is extracted and transformed using TF-IDF vectorization.
3. **Classification**: The SVC model classifies the resume into one of the four predefined categories (ReactJS, Peoplesoft, SQL, Workday), or 'Other' if no suitable category is identified.
4. **Threshold Tuning**: Users can adjust a slider to set a probability threshold for classification to reduce misclassification.
5. **Folder Creation and Sorting**: Based on the classification, subfolders are created for each category, and resumes are automatically placed into the appropriate folder.
6. **CSV Output**: Users can download a CSV file containing the filename and its assigned category.

### Usage

1. Upload a resume in either `.pdf` or `.docx` format.
2. Adjust the threshold slider to set the confidence level for classification.
3. View the classification results indicating which category the resume belongs to, or if it is classified as 'Other' based on the threshold.
4. Download the CSV file listing the filenames and their corresponding categories.
5. Check the newly created folders in the system where resumes are automatically sorted into subfolders based on their category.

### Model Details

- **Model**: Support Vector Classifier (SVC)
- **Text Feature Extraction**: TF-IDF
- **Fine-Tuning**: GridSearchCV
- **Classification Output**: ReactJS, Peoplesoft, SQL, Workday, Other (if below threshold)

### Performance

The model's performance varies depending on the classification threshold. For default settings, it provides high precision in categorizing resumes that fit the four predefined categories.

### Limitations

- The model classifies resumes into one of the four categories or 'Other', so resumes with skillsets outside these categories may still be misclassified if the threshold is not set appropriately.
- The model currently supports only `.pdf` and `.docx` formats.

### Future Work

- Deployment of the model as a web application for public access.
- Expanding the number of categories.
- Improving the classification algorithm to handle resumes with multiple relevant skillsets.

### Contributing

Contributions are welcome! If you have any suggestions or want to contribute to this project, please create a pull request.
