import React, { useState} from "react";

// ====================== MOCK AUTH ======================
const mockUser = {
  display_name: "John Doe",
  username: "johndoe",
  role: "patient",
};

// ====================== DASHBOARD ======================
const menuItems = [
  { id: "consult", label: "Consult Doctor", icon: "👨‍⚕️" },
  { id: "myConsultations", label: "My Consultations", icon: "💬" },
  { id: "forum", label: "Community Forum", icon: "🌐" },
  { id: "cases", label: "My Cases", icon: "📚" },
  { id: "notifications", label: "Notifications", icon: "🔔" },
  { id: "settings", label: "Settings", icon: "⚙️" },
];

export default function PatientDashboard() {
  const [active, setActive] = useState("consult");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // ===== MOCK DATA =====
  const [doctors] = useState([
    { username: "drsmith", display_name: "Dr. Smith", specialization: "Cardiology", online: true },
    { username: "drjones", display_name: "Dr. Jones", specialization: "Neurology", online: true },
  ]);

  const [consultations, setConsultations] = useState([]);
  const [posts, setPosts] = useState([]);
  const [cases, setCases] = useState([]);
  const [notifications] = useState([{ id: 1, text: "Welcome to MedSpatial AI!", time: "Just now", read: false }]);

  return (
    <div className="flex h-screen bg-gray-50 font-syne">
      {/* SIDEBAR */}
      <div className={`${sidebarCollapsed ? 'w-20' : 'w-72'} bg-white border-r border-gray-100 flex flex-col transition-all duration-300 shadow-sm`}>
        <div className="p-5 border-b border-gray-100 flex items-center justify-between">
          {!sidebarCollapsed && (
            <div>
              <h1 className="font-bold text-lg bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                
              </h1>
              <p className="text-xs text-gray-400 mt-0.5">Patient Panel</p>
            </div>
          )}
          <button onClick={() => setSidebarCollapsed(!sidebarCollapsed)} className="text-gray-400 hover:text-gray-600 p-1">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={sidebarCollapsed ? "M13 5l7 7-7 7M5 5l7 7-7 7" : "M11 19l-7-7 7-7M19 19l-7-7 7-7"} />
            </svg>
          </button>
        </div>

        <div className="flex-1 p-3 space-y-1 overflow-y-auto">
          {menuItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setActive(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm transition-all duration-200 
                ${active === item.id
                  ? "bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-md shadow-blue-600/20"
                  : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                }`}
            >
              <span className="text-lg">{item.icon}</span>
              {!sidebarCollapsed && <span className="font-medium">{item.label}</span>}
            </button>
          ))}
        </div>

        {/* User Profile */}
        <div className="p-4 border-t border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-white text-sm font-bold">
              {(mockUser.display_name || "P")[0].toUpperCase()}
            </div>
            {!sidebarCollapsed && (
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">{mockUser.display_name}</p>
                <p className="text-xs text-gray-400 truncate">@{mockUser.username}</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-8 max-w-6xl mx-auto">
          {active === "consult" && <ConsultDoctor doctors={doctors} consultations={consultations} setConsultations={setConsultations} />}
          {active === "myConsultations" && <MyConsultations consultations={consultations} />}
          {active === "forum" && <CommunityForum posts={posts} setPosts={setPosts} />}
          {active === "cases" && <MyCases cases={cases} setCases={setCases} />}
          {active === "notifications" && <Notifications notifications={notifications} />}
          {active === "settings" && <Settings user={mockUser} />}
        </div>
      </div>
    </div>
  );
}

// ====================== CONSULT DOCTOR ======================
function ConsultDoctor({ doctors, consultations, setConsultations }) {
  const [activeConsultation, setActiveConsultation] = useState(null);

  function startConsultation(doc) {
    const existing = consultations.find(c => c.doctor_username === doc.username);
    if (existing) {
      setActiveConsultation(existing);
    } else {
      const newConsult = {
        id: Date.now(),
        doctor_name: doc.display_name,
        doctor_username: doc.username,
        status: "active",
        messages: [],
      };
      setConsultations([...consultations, newConsult]);
      setActiveConsultation(newConsult);
    }
  }

  if (activeConsultation) {
    return (
      <div>
        <button onClick={() => setActiveConsultation(null)} className="mb-4 text-sm text-gray-500 hover:text-gray-700">
          ← Back to doctors
        </button>
        <div className="bg-white p-6 rounded-xl border border-gray-100">Chat with {activeConsultation.doctor_name} (mock)</div>
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Consult a Doctor</h1>
      <div className="grid md:grid-cols-2 gap-4">
        {doctors.map((doc) => (
          <div key={doc.username} className="bg-white p-6 rounded-2xl border border-gray-100 hover:shadow-lg transition">
            <h3 className="font-semibold">{doc.display_name}</h3>
            <p className="text-sm text-blue-600">{doc.specialization}</p>
            <p className="text-xs mt-1">{doc.online ? "Available" : "Offline"}</p>
            <button onClick={() => startConsultation(doc)} className="mt-4 w-full bg-blue-600 text-white py-2 rounded-xl text-sm">
              Start Consultation
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

// ====================== MY CONSULTATIONS ======================
function MyConsultations({ consultations }) {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">My Consultations</h1>
      {consultations.length === 0 ? (
        <p>No consultations yet.</p>
      ) : (
        <div className="space-y-3">
          {consultations.map((c) => (
            <div key={c.id} className="bg-white p-5 rounded-2xl border border-gray-100">
              <h3 className="font-semibold">{c.doctor_name}</h3>
              <p>Status: {c.status}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ====================== COMMUNITY FORUM ======================
function CommunityForum({ posts, setPosts }) {
  const [showCreate, setShowCreate] = useState(false);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");

  function createPost(e) {
    e.preventDefault();
    if (!title || !content) return;
    const newPost = { id: Date.now(), title, content, comments: [] };
    setPosts([newPost, ...posts]);
    setTitle(""); setContent(""); setShowCreate(false);
  }

  return (
    <div>
      <div className="flex justify-between mb-4">
        <h1 className="text-2xl font-bold">Community Forum</h1>
        <button onClick={() => setShowCreate(!showCreate)} className="bg-blue-600 text-white px-4 py-2 rounded-xl">+ New Post</button>
      </div>
      {showCreate && (
        <form onSubmit={createPost} className="mb-6 space-y-3 p-4 border rounded-xl bg-white">
          <input value={title} onChange={e => setTitle(e.target.value)} placeholder="Title" className="w-full border px-3 py-2 rounded-xl" />
          <textarea value={content} onChange={e => setContent(e.target.value)} placeholder="Content" className="w-full border px-3 py-2 rounded-xl"></textarea>
          <button type="submit" className="bg-blue-600 text-white px-4 py-2 rounded-xl">Post</button>
        </form>
      )}
      <div className="space-y-3">
        {posts.map(p => (
          <div key={p.id} className="bg-white p-4 border rounded-xl">{p.title}: {p.content}</div>
        ))}
      </div>
    </div>
  );
}

// ====================== MY CASES ======================
function MyCases({ cases, setCases }) {
  const [showCreate, setShowCreate] = useState(false);
  const [title, setTitle] = useState("");

  function addCase(e) {
    e.preventDefault();
    if (!title) return;
    const newCase = { id: Date.now(), title, status: "pending", result: "Mock Result", confidence: "90%" };
    setCases([newCase, ...cases]);
    setTitle(""); setShowCreate(false);
  }

  return (
    <div>
      <div className="flex justify-between mb-4">
        <h1 className="text-2xl font-bold">My Cases</h1>
        <button onClick={() => setShowCreate(!showCreate)} className="bg-blue-600 text-white px-4 py-2 rounded-xl">+ Upload Scan</button>
      </div>
      {showCreate && (
        <form onSubmit={addCase} className="mb-6 space-y-3 p-4 border rounded-xl bg-white">
          <input value={title} onChange={e => setTitle(e.target.value)} placeholder="Case Title" className="w-full border px-3 py-2 rounded-xl" />
          <button type="submit" className="bg-blue-600 text-white px-4 py-2 rounded-xl">Upload & Analyze</button>
        </form>
      )}
      {cases.map(c => (
        <div key={c.id} className="bg-white p-4 border rounded-xl mb-2">
          <h3>{c.title}</h3>
          <p>Status: {c.status}</p>
          <p>Result: {c.result} ({c.confidence})</p>
        </div>
      ))}
    </div>
  );
}

// ====================== NOTIFICATIONS ======================
function Notifications({ notifications }) {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Notifications</h1>
      {notifications.map(n => (
        <div key={n.id} className="bg-white p-4 border rounded-xl mb-2">{n.text}</div>
      ))}
    </div>
  );
}

// ====================== SETTINGS ======================
function Settings({ user }) {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Settings</h1>
      <div className="bg-white p-6 rounded-xl border max-w-lg space-y-4">
        <div><label className="text-sm text-gray-500 block mb-1">Display Name</label><input value={user.display_name} className="w-full border px-3 py-2 rounded-xl" readOnly /></div>
        <div><label className="text-sm text-gray-500 block mb-1">Username</label><input value={user.username} className="w-full border px-3 py-2 rounded-xl" readOnly /></div>
        <div><label className="text-sm text-gray-500 block mb-1">Role</label><input value={user.role} className="w-full border px-3 py-2 rounded-xl" readOnly /></div>
      </div>
    </div>
  );
}