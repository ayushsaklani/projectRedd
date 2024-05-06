import sys
import subprocess
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon

class LoginWindow(QWidget):
    def __init__(self, main_app, parent=None):
        super().__init__(parent)
        self.main_app = main_app
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle("Login - REDD")
        self.setGeometry(100, 100, 280, 110)
        self.setFixedSize(500, 500)  # Set fixed size for the login window

        layout = QVBoxLayout()

        intro_label = QLabel("Welcome to REDD - Doctor's portal \n Please enter your credentials to login")  # Introduction text
        intro_label.setAlignment(Qt.AlignCenter)  # Center align the text
        layout.addWidget(intro_label)

        self.username = QLineEdit(self)
        self.username.setPlaceholderText("Username")
        layout.addWidget(self.username)

        self.password = QLineEdit(self)
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setPlaceholderText("Password")
        layout.addWidget(self.password)

        loginButton = QPushButton('Login', self)
        loginButton.clicked.connect(self.check_credentials)
        layout.addWidget(loginButton)

        self.setLayout(layout)

    def check_credentials(self):
        username = self.username.text()
        password = self.password.text()

        with open("credentials.txt", "r") as file:
            for line in file:
                user, passw = line.strip().split(",")
                if user == username and passw == password:
                    self.main_app.show()
                    self.close()
                    return
        QLabel("Login Failed", self).show()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'REDD'
        self.left = 100
        self.top = 100
        self.width = 600
        self.height = 500
        self.img_directory = "/Users/namhokoh/desktop"  # Path to the directory where images are stored
        self.initUI()
        self.hide()  # Initially hide the main window

    def initUI(self):
        self.setFixedSize(500, 500)  # Set fixed size for the login window
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowIcon(QIcon('app_icon.png'))  # Set the icon of the app
 
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
 
        self.setupBody()
        self.setupFooter()
 
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-size: 14px;
            }
            QPushButton {
                background-color: #0078D7;
                color: white;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #0053ba;
            }
            QLabel {
                color: #333333;
            }
        """)
 
    def setupBody(self):
        self.imageLabel = QLabel('Upload an image to start', self)
        self.layout.addWidget(self.imageLabel)

    def setupFooter(self):
        footerLayout = QVBoxLayout()

        uploadButton = QPushButton('Upload patient scan', self)
        uploadButton.clicked.connect(self.upload_image)
        footerLayout.addWidget(uploadButton)

        runButton = QPushButton('Run Inference', self)
        runButton.clicked.connect(self.run_inference)
        footerLayout.addWidget(runButton)

        self.layout.addLayout(footerLayout)

    def upload_image(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if self.file_path:
            pixmap = QPixmap(self.file_path)
            self.imageLabel.setPixmap(pixmap.scaled(560, 315, Qt.KeepAspectRatio))
        else:
            self.imageLabel.setText('Upload an image to start')
            
    def display_results(self):
        processed_image_path = self.file_path.replace('.png', '_processed.png').replace('.jpg', '_processed.jpg').replace('.jpeg', '_processed.jpeg')
        
        # Debug print
        print(f"Looking for processed image at: {processed_image_path}")

        if os.path.exists(processed_image_path):
            pixmap = QPixmap(processed_image_path)
            self.imageLabel.setPixmap(pixmap.scaled(560, 315, Qt.KeepAspectRatio))
        else:
            self.imageLabel.setText('Processed image not found.')        
 
    def run_inference(self):
        if self.file_path:
            cmd = f"python detect.py --image_path \"{self.file_path}\""
            subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).wait()
            self.display_results()
 
    def get_latest_folder(self, path):
        folders = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if not folders:
            return None
        latest_folder = max(folders, key=os.path.getmtime)
        return latest_folder
 
    def get_latest_image(self, folder_path):
        jpg_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]
        if not jpg_files:
            return None
        latest_image = max(jpg_files, key=os.path.getmtime)
        return latest_image

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = App()
    login_window = LoginWindow(main_app)
    login_window.show()
    sys.exit(app.exec_())