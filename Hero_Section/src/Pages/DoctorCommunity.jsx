// import React, { useState } from "react";

// const menuItems = [
//   { id: "dashboard", label: "Dashboard", icon: "🏠" },
//   { id: "incoming", label: "Incoming Cases", icon: "📥" },
//   { id: "aiCases", label: "AI Flagged Cases", icon: "🧠" },
//   { id: "consultations", label: "Consultations", icon: "💬" },
//   { id: "history", label: "Patient History", icon: "📁" },
//   { id: "reports", label: "Reports & Notes", icon: "📝" },
//   { id: "forum", label: "Community Forum", icon: "🌐" },
//   { id: "analytics", label: "Analytics", icon: "📊" },
//   { id: "settings", label: "Settings", icon: "⚙️" },
// ];

// const incomingCasesData = [
//   {
//     id: 1,
//     patient: "Rohit Sharma",
//     type: "Chest CT",
//     status: "Pending",
//     priority: "Normal",
//   },
//   {
//     id: 2,
//     patient: "Ananya Verma",
//     type: "Lung CT",
//     status: "Pending",
//     priority: "Urgent",
//   },
// ];

// function IncomingCases() {
//   return (
//     <div>
//       <h1 className="text-xl font-semibold mb-4">Incoming Cases</h1>

//       <div className="space-y-3">
//         {incomingCasesData.map((c) => (
//           <div
//             key={c.id}
//             className="p-4 bg-white rounded shadow flex justify-between items-center"
//           >
//             <div>
//               <h3 className="font-medium">{c.patient}</h3>
//               <p className="text-sm text-gray-500">{c.type}</p>
//             </div>

//             <span
//               className={`text-xs px-3 py-1 rounded ${
//                 c.priority === "Urgent"
//                   ? "bg-red-500 text-white"
//                   : "bg-gray-200"
//               }`}
//             >
//               {c.priority}
//             </span>
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// }

// const aiCasesData = [
//   {
//     id: 1,
//     patient: "Vikas Patel",
//     issue: "Possible Lung Nodule",
//     confidence: "91%",
//   },
//   {
//     id: 2,
//     patient: "Sneha Iyer",
//     issue: "Opacity Detected",
//     confidence: "84%",
//   },
// ];

// function AICases() {
//   return (
//     <div>
//       <h1 className="text-xl font-semibold mb-4">AI Flagged Cases</h1>

//       <div className="space-y-3">
//         {aiCasesData.map((c) => (
//           <div key={c.id} className="p-4 bg-red-100 rounded shadow">
//             <h3 className="font-medium">{c.patient}</h3>
//             <p className="text-sm">{c.issue}</p>
//             <p className="text-xs text-gray-600">
//               Confidence: {c.confidence}
//             </p>
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// }

// const consultationsData = [
//   {
//     id: 1,
//     patient: "Rohit Sharma",
//     lastMessage: "Doctor, is this serious?",
//   },
//   {
//     id: 2,
//     patient: "Ananya Verma",
//     lastMessage: "Uploaded my scan, please check",
//   },
// ];

// function Consultations() {
//   return (
//     <div>
//       <h1 className="text-xl font-semibold mb-4">Consultations</h1>

//       <div className="space-y-3">
//         {consultationsData.map((c) => (
//           <div key={c.id} className="p-3 bg-white rounded shadow">
//             <h3 className="font-medium">{c.patient}</h3>
//             <p className="text-sm text-gray-500">{c.lastMessage}</p>
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// }

// const historyData = [
//   {
//     id: 1,
//     patient: "Rohit Sharma",
//     cases: 3,
//   },
//   {
//     id: 2,
//     patient: "Ananya Verma",
//     cases: 2,
//   },
// ];

// function PatientHistory() {
//   return (
//     <div>
//       <h1 className="text-xl font-semibold mb-4">Patient History</h1>

//       {historyData.map((p) => (
//         <div key={p.id} className="p-3 bg-white rounded shadow mb-2">
//           <h3>{p.patient}</h3>
//           <p className="text-sm text-gray-500">
//             Total Cases: {p.cases}
//           </p>
//         </div>
//       ))}
//     </div>
//   );
// }



// export default function DoctorDashboard() {
//   const [active, setActive] = useState("dashboard");

//   return (
//     <div className="flex h-screen bg-gray-100">

//       {/* SIDEBAR */}
//       <div className="w-64 bg-white border-r flex flex-col">
//         <div className="p-5 font-bold text-lg border-b">
//           MedSpatial AI (Doctor)
//         </div>

//         <div className="flex-1 p-3 space-y-2 overflow-y-auto">
//           {menuItems.map((item) => (
//             <button
//               key={item.id}
//               onClick={() => setActive(item.id)}
//               className={`w-full flex items-center gap-3 px-4 py-2 rounded-lg text-sm transition
//                 ${
//                   active === item.id
//                     ? "bg-black text-white"
//                     : "text-gray-700 hover:bg-gray-200"
//                 }`}
//             >
//               <span>{item.icon}</span>
//               {item.label}

//               {/* Example badge */}
//               {item.id === "incoming" && (
//                 <span className="ml-auto text-xs bg-red-500 text-white px-2 rounded-full">
//                   3
//                 </span>
//               )}
//             </button>
//           ))}
//         </div>

//         <div className="p-4 text-xs text-gray-400 border-t">
//           Doctor Panel
//         </div>
//       </div>

//       {/* MAIN CONTENT */}
//       <div className="flex-1 p-6 overflow-y-auto">

//         {active === "dashboard" && <Dashboard />}
//         {active === "incoming" && <IncomingCases />}
//         {active === "aiCases" && <AICases />}
//         {active === "consultations" && <Consultations />}
//         {active === "history" && <PatientHistory />}
//         {active === "reports" && <Reports />}
//         {active === "forum" && <Forum />}
//         {active === "analytics" && <Analytics />}
//         {active === "settings" && <Settings />}

//       </div>
//     </div>
//   );
// }

// /////////////////////////////////////////////////////
// // PAGES (STARTER VERSIONS)
// /////////////////////////////////////////////////////

// function Dashboard() {
//   return (
//     <div>
//       <h1 className="text-xl font-semibold mb-4">Dashboard</h1>
//       <p className="text-gray-600">Overview of activity and workload.</p>
//     </div>
//   );
// }

// // function IncomingCases() {
// //   return (
// //     <div>
// //       <h1 className="text-xl font-semibold mb-4">Incoming Cases</h1>
// //       <p className="text-gray-600">New patient uploads waiting for review.</p>

// //       <div className="mt-4 space-y-3">
// //         <div className="p-3 bg-white rounded shadow">
// //           Lung CT - Pending Review
// //         </div>
// //         <div className="p-3 bg-white rounded shadow">
// //           Chest Scan - Urgent
// //         </div>
// //       </div>
// //     </div>
// //   );
// // }

// // function AICases() {
// //   return (
// //     <div>
// //       <h1 className="text-xl font-semibold mb-4">AI Flagged Cases</h1>
// //       <p className="text-gray-600">High-risk cases detected by AI.</p>

// //       <div className="mt-4 p-3 bg-red-100 rounded">
// //         Possible tumor detected (Confidence: 91%)
// //       </div>
// //     </div>
// //   );
// // }

// // function Consultations() {
// //   return (
// //     <div>


// //       <h1 className="text-xl font-semibold mb-4">Live Consultations</h1>
// //       <p className="text-gray-600">Active chats with patients.</p>
// //     </div>
// //   );
// // }

// // function PatientHistory() {
// //   return (
// //     <div>
// //       <h1 className="text-xl font-semibold mb-4">Patient History</h1>
// //       <p className="text-gray-600">View past scans and reports.</p>
// //     </div>
// //   );
// // }

// function Reports() {
//   return (
//     <div>
//       <h1 className="text-xl font-semibold mb-4">Reports & Notes</h1>

//       <textarea
//         defaultValue="Patient shows signs of mild lung infection. Recommend antibiotics and follow-up scan."
//         className="w-full h-40 border p-3 rounded"
//       />

//       <button className="mt-3 px-4 py-2 bg-black text-white rounded">
//         Save Report
//       </button>
//     </div>
//   );
// }

// function Forum() {
//   return (
//     <div>
//       <h1 className="text-xl font-semibold mb-4">Community Forum</h1>
//       <p className="text-gray-600">Discuss cases with other doctors.</p>
//     </div>
//   );
// }

// function Analytics() {
//   return (
//     <div>
//       <h1 className="text-xl font-semibold mb-4">Analytics</h1>

//       <div className="grid grid-cols-3 gap-4">
//         <div className="bg-white p-4 rounded shadow">
//           <p className="text-sm text-gray-500">Cases Reviewed</p>
//           <h2 className="text-xl font-bold">128</h2>
//         </div>

//         <div className="bg-white p-4 rounded shadow">
//           <p className="text-sm text-gray-500">AI Flagged</p>
//           <h2 className="text-xl font-bold">32</h2>
//         </div>

//         <div className="bg-white p-4 rounded shadow">
//           <p className="text-sm text-gray-500">Consultations</p>
//           <h2 className="text-xl font-bold">54</h2>
//         </div>
//       </div>
//     </div>
//   );
// }

// function Settings() {
//   return (
//     <div>
//       <h1 className="text-xl font-semibold mb-4">Settings</h1>
//       <p className="text-gray-600">Manage account preferences.</p>
//     </div>
//   );
// }