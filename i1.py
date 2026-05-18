import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox
import io
import base64
from PIL import Image, ImageTk

def safe_insert(text_widget, msg):
    text_widget.insert(tk.END, msg)
    text_widget.see(tk.END)

def is_dataframe_loaded(df):
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty

class SDNAnomalyDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Encrypted SDN Traffic Anomaly Detection System")
        master.geometry("1200x800")
        
        btn_cfg = {'width': 25, 'bg': 'lightgreen', 'activebackground': '#bfeecf', 'font': ('Arial', 9)}
        
        # Title
        title_frame = tk.Frame(master, bg='lightblue')
        title_frame.pack(fill=tk.X, pady=(5, 10))
        self.title_label = tk.Label(title_frame, text="🔒 Encrypted SDN Traffic Anomaly Detection Dashboard", 
                                   font=('Helvetica', 18, 'bold'), bg='lightblue', fg='darkblue')
        self.title_label.pack(pady=10)
        
        # Main frames
        self.top_frame = tk.Frame(master)
        self.top_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.left_frame = tk.Frame(self.top_frame, bg='lightgray')
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.left_frame.pack_propagate(False)
        self.left_frame.configure(width=300)
        
        self.right_frame = tk.Frame(self.top_frame, bg='white')
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Control Panel Header
        tk.Label(self.left_frame, text="CONTROL PANEL", font=('Arial', 12, 'bold'), 
                bg='lightgray', fg='darkblue').pack(pady=(10,5))
        
        # Load Data Buttons
        load_frame = tk.LabelFrame(self.left_frame, text="1. Load Data", font=('Arial', 10, 'bold'), padx=5, pady=5)
        load_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.load_train_btn = tk.Button(load_frame, text="📁 Load Train CSV", command=self.load_train, **btn_cfg)
        self.load_train_btn.pack(pady=2, fill=tk.X)
        
        self.load_test_btn = tk.Button(load_frame, text="📁 Load Test CSV", command=self.load_test, **btn_cfg)
        self.load_test_btn.pack(pady=2, fill=tk.X)
        
        # Processing Buttons
        proc_frame = tk.LabelFrame(self.left_frame, text="2. Process", font=('Arial', 10, 'bold'), padx=5, pady=5)
        proc_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.preprocess_btn = tk.Button(proc_frame, text="🔄 Preprocess Data", command=self.preprocess, **btn_cfg)
        self.preprocess_btn.pack(pady=2, fill=tk.X)
        
        self.train_btn = tk.Button(proc_frame, text="🚀 Train Model", command=self.train_model, **btn_cfg)
        self.train_btn.pack(pady=2, fill=tk.X)
        
        # EDA Buttons
        eda_frame = tk.LabelFrame(self.left_frame, text="3. EDA Charts", font=('Arial', 10, 'bold'), padx=5, pady=5)
        eda_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.eda1_btn = tk.Button(eda_frame, text="📊 Attack Distribution", command=self.show_eda_attacks, **btn_cfg)
        self.eda1_btn.pack(pady=1, fill=tk.X)
        
        self.eda2_btn = tk.Button(eda_frame, text="📈 Byte Count Dist", command=self.show_eda_bytes, **btn_cfg)
        self.eda2_btn.pack(pady=1, fill=tk.X)
        
        self.eda3_btn = tk.Button(eda_frame, text="🔥 Correlation Heatmap", command=self.show_eda_heatmap, **btn_cfg)
        self.eda3_btn.pack(pady=1, fill=tk.X)
        
        # Prediction Section
        pred_frame = tk.LabelFrame(self.left_frame, text="4. Predict", font=('Arial', 10, 'bold'), padx=5, pady=5)
        pred_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(pred_frame, text="Test Row ID (0-1999):").pack(anchor='w')
        self.row_entry = tk.Entry(pred_frame, width=15, font=('Courier', 12))
        self.row_entry.pack(pady=(0,5), fill=tk.X)
        self.row_entry.insert(0, "0")
        
        self.predict_btn = tk.Button(pred_frame, text="🎯 Predict Attack", command=self.predict_row, **btn_cfg)
        self.predict_btn.pack(pady=2, fill=tk.X)
        
        self.clear_output_btn = tk.Button(pred_frame, text="🗑️ Clear Output", command=self.clear_output, bg='lightcoral')
        self.clear_output_btn.pack(pady=2, fill=tk.X)
        
        # Output Area
        output_frame = tk.LabelFrame(self.right_frame, text="OUTPUT & RESULTS", font=('Arial', 12, 'bold'), padx=10, pady=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable text output
        text_frame = tk.Frame(output_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.output_box = tk.Text(text_frame, height=25, width=80, bg='#f0f0f0', fg='black', 
                                 font=('Courier', 10), wrap=tk.WORD)
        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.output_box.yview)
        self.output_box.configure(yscrollcommand=scrollbar.set)
        
        self.output_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load your train/test CSV files to begin!")
        self.status_bar = tk.Label(master, textvariable=self.status_var, bd=2, relief=tk.SUNKEN, 
                                  anchor=tk.W, bg='lightblue', fg='darkblue', font=('Arial', 10))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Data/model holders
        self.train_data = None
        self.test_data = None
        self.model = None
        self.label_encoders = {}
        
    def update_status(self, msg):
        self.status_var.set(msg)
        self.master.update()
        
    def load_train(self):
        try:
            filepath = filedialog.askopenfilename(title="Select Train CSV File", 
                                                filetypes=[("CSV files", "*.csv")])
            if not filepath:
                return
            self.train_data = pd.read_csv(filepath)
            safe_insert(self.output_box, f"✅ TRAINING DATA LOADED\n")
            safe_insert(self.output_box, f"   Rows: {self.train_data.shape[0]:,} | Columns: {self.train_data.shape[1]}\n")
            safe_insert(self.output_box, f"   Columns: {list(self.train_data.columns)}\n")
            safe_insert(self.output_box, f"   Unique labels: {self.train_data['label'].nunique() if 'label' in self.train_data.columns else 'N/A'}\n")
            safe_insert(self.output_box, "-"*60 + "\n\n")
            self.update_status(f"Training data loaded: {self.train_data.shape[0]:,} rows")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load training file:\n{str(e)}")
            
    def load_test(self):
        try:
            filepath = filedialog.askopenfilename(title="Select Test CSV File", 
                                                filetypes=[("CSV files", "*.csv")])
            if not filepath:
                return
            self.test_data = pd.read_csv(filepath)
            safe_insert(self.output_box, f"✅ TESTING DATA LOADED\n")
            safe_insert(self.output_box, f"   Rows: {self.test_data.shape[0]:,} | Columns: {self.test_data.shape[1]}\n")
            safe_insert(self.output_box, "-"*60 + "\n\n")
            self.update_status(f"Test data loaded: {self.test_data.shape[0]:,} rows")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load testing file:\n{str(e)}")
            
    def preprocess(self):
        try:
            if not is_dataframe_loaded(self.train_data) or not is_dataframe_loaded(self.test_data):
                messagebox.showerror("Error", "Please load both train and test CSV files first!")
                return
            
            self.update_status("Preprocessing data...")
            
            # Clean numeric columns
            numeric_cols = ['packet_count', 'byte_count', 'duration', 'avg_packet_size']
            for df in [self.train_data, self.test_data]:
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
            
            # Categorical columns specific to SDN flows
            cat_cols = ['protocol', 'direction']
            target_col = 'label'
            
            # Encode categorical columns
            for col in cat_cols:
                if col in self.train_data.columns:
                    le = LabelEncoder()
                    self.train_data[col] = le.fit_transform(self.train_data[col].astype(str))
                    self.label_encoders[col] = le
                    
                    if col in self.test_data.columns:
                        try:
                            self.test_data[col] = le.transform(self.test_data[col].astype(str))
                        except:
                            mapping = {v: i for i, v in enumerate(le.classes_)}
                            self.test_data[col] = self.test_data[col].astype(str).map(mapping).fillna(-1)
            
            # Encode target variable
            if target_col in self.train_data.columns:
                le_target = LabelEncoder()
                self.train_data[target_col] = le_target.fit_transform(self.train_data[target_col].astype(str))
                self.label_encoders[target_col] = le_target
                
                if target_col in self.test_data.columns:
                    try:
                        self.test_data[target_col] = le_target.transform(self.test_data[target_col].astype(str))
                    except:
                        mapping = {v: i for i, v in enumerate(le_target.classes_)}
                        self.test_data[target_col] = self.test_data[target_col].astype(str).map(mapping).fillna(-1)
            
            safe_insert(self.output_box, "✅ PREPROCESSING COMPLETE\n")
            safe_insert(self.output_box, f"   Encoded: {cat_cols + [target_col]}\n")
            safe_insert(self.output_box, f"   Numeric cleaned: {numeric_cols}\n")
            safe_insert(self.output_box, "-"*60 + "\n\n")
            self.update_status("Preprocessing finished - Ready to train!")
            
        except Exception as e:
            messagebox.showerror("Preprocess Error", f"Preprocessing failed:\n{str(e)}")
            
    def train_model(self):
        try:
            if not is_dataframe_loaded(self.train_data):
                messagebox.showerror("Error", "Training data not loaded!")
                return
                
            if 'label' not in self.train_data.columns:
                messagebox.showerror("Error", "Target column 'label' not found!")
                return
            
            self.update_status("Training Random Forest model...")
            
            # SDN-specific feature engineering (drop identifiers)
            feature_cols = ['src_port', 'dst_port', 'protocol', 'packet_count', 'byte_count', 
                           'duration', 'avg_packet_size', 'direction']
            available_features = [c for c in feature_cols if c in self.train_data.columns]
            
            X_train = self.train_data[available_features]
            y_train = self.train_data['label']
            
            # Multi-class anomaly detection model
            clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced',
                                       max_depth=15, min_samples_split=10)
            clf.fit(X_train, y_train)
            self.model = clf
            
            # Evaluate on training data
            preds = clf.predict(X_train)
            acc = accuracy_score(y_train, preds)
            cm = confusion_matrix(y_train, preds)
            
            safe_insert(self.output_box, "✅ MODEL TRAINING COMPLETE\n")
            safe_insert(self.output_box, f"   Algorithm: Random Forest (200 trees)\n")
            safe_insert(self.output_box, f"   Training Accuracy: {acc:.4f}\n")
            safe_insert(self.output_box, f"   Attack Classes: {len(np.unique(y_train))}\n")
            safe_insert(self.output_box, f"   Features Used: {len(available_features)}\n")
            safe_insert(self.output_box, "   Confusion Matrix:\n")
            safe_insert(self.output_box, str(cm) + "\n")
            safe_insert(self.output_box, "-"*60 + "\n\n")
            
            # Feature importance
            importances = clf.feature_importances_
            for i, feat in enumerate(available_features):
                safe_insert(self.output_box, f"   {feat:15}: {importances[i]:.4f}\n")
            safe_insert(self.output_box, "-"*60 + "\n\n")
            
            self.update_status(f"Model trained! Accuracy: {acc:.1%}")
            
        except Exception as e:
            messagebox.showerror("Training Error", f"Training failed:\n{str(e)}")
            
    def predict_row(self):
        try:
            if self.model is None:
                messagebox.showerror("Error", "No model available! Train first.")
                return
                
            if not is_dataframe_loaded(self.test_data):
                messagebox.showerror("Error", "Test data not loaded!")
                return
                
            row_id = int(self.row_entry.get())
            if row_id < 0 or row_id >= self.test_data.shape[0]:
                messagebox.showerror("Error", f"Row ID must be 0 to {self.test_data.shape[0]-1}")
                return
            
            feature_cols = ['src_port', 'dst_port', 'protocol', 'packet_count', 'byte_count', 
                           'duration', 'avg_packet_size', 'direction']
            available_features = [c for c in feature_cols if c in self.test_data.columns]
            
            input_row = self.test_data[available_features].iloc[row_id:row_id+1]
            pred = self.model.predict(input_row)[0]
            pred_proba = self.model.predict_proba(input_row)[0]
            
            le_target = self.label_encoders.get('label', None)
            pred_decoded = le_target.inverse_transform([pred])[0] if le_target else str(pred)
            
            safe_insert(self.output_box, f"\n🎯 PREDICTION FOR ROW {row_id}\n")
            safe_insert(self.output_box, "="*50 + "\n")
            
            for col in available_features:
                val = self.test_data[col].iloc[row_id]
                if col in self.label_encoders:
                    try:
                        val = self.label_encoders[col].inverse_transform([int(val)])[0]
                    except:
                        pass
                safe_insert(self.output_box, f"  {col:15}: {val}\n")
            
            safe_insert(self.output_box, f"\n🔴 PREDICTED ATTACK: {pred_decoded}\n")
            safe_insert(self.output_box, f"   Confidence: {max(pred_proba):.1%}\n")
            safe_insert(self.output_box, "-"*50 + "\n\n")
            
            self.update_status(f"Predicted row {row_id}: {pred_decoded}")
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Prediction failed:\n{str(e)}")
            
    def show_eda_attacks(self):
        try:
            if not is_dataframe_loaded(self.train_data):
                messagebox.showerror("Error", "Load training data first!")
                return
                
            win = tk.Toplevel(self.master)
            win.title("📊 Attack Distribution Analysis")
            win.geometry("800x600")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            data = self.train_data.copy()
            if 'label' in self.label_encoders:
                try:
                    data['label'] = self.label_encoders['label'].inverse_transform(data['label'].astype(int))
                except:
                    pass
                    
            order = data['label'].value_counts().index
            sns.countplot(data=data, x='label', palette='viridis', ax=ax, order=order)
            ax.set_title("SDN Attack Distribution (Training Data)", fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Attack Type", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("EDA Error", str(e))
            
    def show_eda_bytes(self):
        try:
            if not is_dataframe_loaded(self.train_data):
                messagebox.showerror("Error", "Load training data first!")
                return
                
            win = tk.Toplevel(self.master)
            win.title("📈 Byte Count Distribution")
            win.geometry("800x600")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(data=self.train_data, x='byte_count', kde=True, bins=50, ax=ax)
            ax.set_title("Byte Count Distribution Across All Flows", fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Byte Count", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("EDA Error", str(e))
            
    def show_eda_heatmap(self):
        try:
            if not is_dataframe_loaded(self.train_data):
                messagebox.showerror("Error", "Load training data first!")
                return
                
            numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                messagebox.showwarning("Warning", "Need numeric columns for heatmap!")
                return
                
            corr = self.train_data[numeric_cols].corr()
            
            win = tk.Toplevel(self.master)
            win.title("🔥 Feature Correlation Heatmap")
            win.geometry("900x700")
            
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, 
                       square=True, cbar_kws={'shrink': 0.8}, ax=ax, linewidths=0.5)
            ax.set_title("SDN Flow Feature Correlation Matrix", fontsize=18, fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("EDA Error", str(e))
            
    def clear_output(self):
        self.output_box.delete(1.0, tk.END)
        self.update_status("Output cleared")

if __name__ == "__main__":
    root = tk.Tk()
    app = SDNAnomalyDetectionApp(root)
    root.mainloop()
