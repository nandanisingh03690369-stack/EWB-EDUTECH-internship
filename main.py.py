import sqlite3
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field
import requests

app = FastAPI(
    title="EWB Students Database CRUD Application",
    description="Backend API built with FastAPI and SQLite for managing student records, with optional Ollama AI integration.",
    version="1.0.0"
)

DB_FILE = "students.db"

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            dob TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            course TEXT,
            address TEXT,
            enrollment_date TEXT
        )
    """)
    conn.commit()
    conn.close()

@app.on_event("startup")
def startup_event():
    init_db()

# --- Pydantic Models ---
class StudentCreate(BaseModel):
    name: str = Field(..., example="John Doe")
    dob: str = Field(..., example="2003-05-14")
    email: EmailStr = Field(..., example="john@example.com")
    phone: str = Field(..., example="+919876543210")
    course: Optional[str] = Field(None, example="Computer Science")
    address: Optional[str] = Field(None, example="123 Main Street")
    enrollment_date: Optional[str] = Field(None, example="2024-08-01")

class StudentUpdate(BaseModel):
    name: Optional[str] = None
    dob: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    course: Optional[str] = None
    address: Optional[str] = None
    enrollment_date: Optional[str] = None

class StudentResponse(StudentCreate):
    student_id: int

class AIQueryRequest(BaseModel):
    prompt: str = Field(..., example="How many students are enrolled in Computer Science?")

# --- Helper Function for DB Connection ---
def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Enables column access by name
    return conn

# --- CRUD Endpoints ---

@post("/students", response_model=StudentResponse, status_code=201)
def create_student(student: StudentCreate):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO students (name, dob, email, phone, course, address, enrollment_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (student.name, student.dob, student.email, student.phone, student.course, student.address, student.enrollment_date))
    conn.commit()
    student_id = cursor.lastrowid
    conn.close()
    
    return {**student.dict(), "student_id": student_id}

@get("/students", response_model=List[StudentResponse])
def list_students(
    course: Optional[str] = Query(None, description="Filter by course name"),
    born_after: Optional[str] = Query(None, description="Filter students born after date (YYYY-MM-DD)")
):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM students WHERE 1=1"
    params = []
    
    if course:
        query += " AND course = ?"
        params.append(course)
    if born_after:
        query += " AND dob > ?"
        params.append(born_after)
        
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

@get("/students/{student_id}", response_model=StudentResponse)
def get_student(student_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Student not found")
    return dict(row)

@put("/students/{student_id}", response_model=StudentResponse)
def update_student(student_id: int, student_update: StudentUpdate):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Student not found")
        
    update_data = student_update.dict(exclude_unset=True)
    if not update_data:
        conn.close()
        raise HTTPException(status_code=400, detail="No fields provided for update")
        
    set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
    values = list(update_data.values()) + [student_id]
    
    cursor.execute(f"UPDATE students SET {set_clause} WHERE student_id = ?", values)
    conn.commit()
    
    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    updated_row = cursor.fetchone()
    conn.close()
    
    return dict(updated_row)

@delete("/students/{student_id}", status_code=204)
def delete_student(student_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Student not found")
        
    cursor.execute("DELETE FROM students WHERE student_id = ?_id", {"_id": student_id}) # standard syntax below
    cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
    conn.commit()
    conn.close()
    return

# --- Optional AI Extension (Ollama Integration) ---
@post("/ai/query")
def ask_ai_about_students(payload: AIQueryRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students")
    students = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # Contextual prompt payload for Ollama
    context_prompt = (
        f"You are a helpful assistant for a student database system. "
        f"Here is the current student database records in JSON format:\n{students}\n\n"
        f"User Question: {payload.prompt}\n"
        f"Provide an accurate, concise, and human-readable answer based strictly on the provided records."
    )
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": context_prompt,
                "stream": False
            },
            timeout=30
        )
        res_json = response.json()
        return {"response": res_json.get("response", "No response generated.")}
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama service is not running locally. Make sure Ollama is active.")