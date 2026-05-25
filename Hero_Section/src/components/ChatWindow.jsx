// import React, { useState, useRef, useEffect } from "react";
// import { useAuth } from "../context/AuthContext";
// import { tables, reducers } from "../module_bindings";

// export default function ChatWindow({ consultation, onClose }) {
//   const { user } = useAuth();
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState("");
//   const [fileName, setFileName] = useState("");
//   const messagesEndRef = useRef(null);
//   const fileInputRef = useRef(null);

//   useEffect(() => {
//     if (!consultation?.id) return;

//     function sync() {
//       const msgs = Array.from(tables.chat_message.iter())
//         .filter(m => m.consultation_id === consultation.id)
//         .sort((a, b) => Number(a.sent_at - b.sent_at));
      
//       setMessages(msgs);
//     }
    
//     sync();

//     const unsubInsert = tables.chat_message.onInsert((ctx, msg) => {
//       if (msg.consultation_id === consultation.id) {
//         sync();
//       }
//     });

//     return () => {
//       // cleanup
//     };
//   }, [consultation]);

//   useEffect(() => {
//     messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
//   }, [messages]);

//   function sendMessage(e) {
//     e.preventDefault();
//     if (!input.trim() && !fileName) return;

//     // Call SpacetimeDB reducer
//     reducers.send_message(
//       consultation.id,
//       input,
//       fileName || null,
//       fileName ? "#" : null
//     );

//     setInput("");
//     setFileName("");
//   }

//   function handleFileSelect(e) {
//     const file = e.target.files?.[0];
//     if (file) {
//       setFileName(file.name);
//     }
//   }

//   function formatTime(ts) {
//     return new Date(Number(ts)).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
//   }

//   const otherName = user?.role === "patient" ? consultation?.doctor_name : consultation?.patient_name;

//   return (
//     <div className="flex flex-col h-[calc(100vh-180px)] bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
//       {/* Header */}
//       <div className="flex items-center justify-between px-6 py-4 border-b bg-gradient-to-r from-gray-50 to-white">
//         <div className="flex items-center gap-3">
//           <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-white font-bold text-sm">
//             {(otherName || "?")[0].toUpperCase()}
//           </div>
//           <div>
//             <h3 className="font-semibold text-gray-900">{otherName || "Consultation"}</h3>
//             <span className="text-xs text-green-500 flex items-center gap-1">
//               <span className="w-2 h-2 bg-green-400 rounded-full inline-block animate-pulse" />
//               Online
//             </span>
//           </div>
//         </div>
//         {onClose && (
//           <button onClick={onClose} className="text-gray-400 hover:text-gray-600 transition p-2 hover:bg-gray-100 rounded-lg">
//             <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
//               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
//             </svg>
//           </button>
//         )}
//       </div>

//       {/* Messages */}
//       <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-gray-50/50">
//         {messages.length === 0 && (
//           <div className="flex flex-col items-center justify-center h-full text-gray-400">
//             <svg className="w-16 h-16 mb-4 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
//               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
//             </svg>
//             <p className="text-sm">Start a conversation</p>
//           </div>
//         )}

//         {messages.map((msg) => {
//           const isMe = msg.sender_name === (user?.display_name || "You");
//           return (
//             <div key={msg.id} className={`flex ${isMe ? "justify-end" : "justify-start"}`}>
//               <div className={`max-w-[70%] ${isMe ? "order-2" : ""}`}>
//                 <div
//                   className={`px-4 py-3 rounded-2xl text-sm shadow-sm ${
//                     isMe
//                       ? "bg-gradient-to-r from-blue-600 to-blue-500 text-white rounded-br-md"
//                       : "bg-white text-gray-800 border border-gray-100 rounded-bl-md"
//                   }`}
//                 >
//                   {!isMe && (
//                     <p className={`text-xs font-semibold mb-1 ${isMe ? "text-blue-100" : "text-blue-600"}`}>
//                       {msg.sender_name}
//                       {msg.sender_role === "doctor" && " 🩺"}
//                     </p>
//                   )}
//                   {msg.text && <p>{msg.text}</p>}
//                   {msg.file_name && (
//                     <div className={`mt-2 flex items-center gap-2 text-xs ${isMe ? "text-blue-100" : "text-gray-500"}`}>
//                       <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
//                         <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
//                       </svg>
//                       📎 {msg.file_name}
//                     </div>
//                   )}
//                 </div>
//                 <p className={`text-[10px] mt-1 ${isMe ? "text-right" : "text-left"} text-gray-400`}>
//                   {formatTime(msg.sent_at)}
//                 </p>
//               </div>
//             </div>
//           );
//         })}
//         <div ref={messagesEndRef} />
//       </div>

//       {/* File indicator */}
//       {fileName && (
//         <div className="px-4 py-2 bg-blue-50 border-t border-blue-100 flex items-center justify-between">
//           <span className="text-sm text-blue-700 flex items-center gap-2">
//             📎 {fileName}
//           </span>
//           <button onClick={() => setFileName("")} className="text-blue-400 hover:text-blue-600 text-xs">
//             Remove
//           </button>
//         </div>
//       )}

//       {/* Input */}
//       <form onSubmit={sendMessage} className="p-4 border-t bg-white flex items-center gap-3">
//         <input type="file" ref={fileInputRef} onChange={handleFileSelect} className="hidden" />
//         <button
//           type="button"
//           onClick={() => fileInputRef.current?.click()}
//           className="text-gray-400 hover:text-blue-500 transition p-2 hover:bg-gray-50 rounded-lg"
//         >
//           <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
//             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
//           </svg>
//         </button>

//         <input
//           type="text"
//           value={input}
//           onChange={(e) => setInput(e.target.value)}
//           placeholder="Type a message..."
//           className="flex-1 bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200 transition"
//         />

//         <button
//           type="submit"
//           className="bg-gradient-to-r from-blue-600 to-blue-500 text-white p-3 rounded-xl hover:from-blue-500 hover:to-blue-400 transition-all shadow-md shadow-blue-600/20"
//         >
//           <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
//             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
//           </svg>
//         </button>
//       </form>
//     </div>
//   );
// }
