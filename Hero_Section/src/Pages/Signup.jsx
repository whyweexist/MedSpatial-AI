// import React, { useState, useEffect } from "react";
// import { Link, useNavigate } from "react-router-dom";
// import { useAuth } from "../context/AuthContext";
// import { reducers } from "../module_bindings";

// export default function Signup() {
//   const [username, setUsername] = useState("");
//   const [password, setPassword] = useState("");
//   const [displayName, setDisplayName] = useState("");
//   const [role, setRole] = useState("patient");
//   const [specialization, setSpecialization] = useState("");
//   const [error, setError] = useState("");
//   const [loading, setLoading] = useState(false);
//   const { signup, user } = useAuth();
//   const navigate = useNavigate();

//   // Listen to SpacetimeDB reducer callbacks
//   useEffect(() => {
//     const unsub = reducers.onRegisterUser((ctx, username, hash, role, dn, spec) => {
//       setLoading(false);
//       if (ctx.event.status === "failed") {
//         setError(ctx.event.callerError || "Signup failed");
//         return;
//       }
//       // On success, AuthContext will auto-catch the new UserRow and set 'user'
//     });
//     return () => {
//       // Typically you'd remove the listener if supported, but this works fine
//     };
//   }, []);

//   // Redirect once user state is populated
//   useEffect(() => {
//     if (user) {
//       if (user.role === "doctor") {
//         navigate("/doctor");
//       } else {
//         navigate("/community");
//       }
//     }
//   }, [user, navigate]);

//   async function hashPassword(pw) {
//     const encoder = new TextEncoder();
//     const data = encoder.encode(pw);
//     const hash = await crypto.subtle.digest("SHA-256", data);
//     return Array.from(new Uint8Array(hash))
//       .map((b) => b.toString(16).padStart(2, "0"))
//       .join("");
//   }

//   async function handleSubmit(e) {
//     e.preventDefault();
//     setError("");
//     setLoading(true);

//     try {
//       if (!username || !password || !displayName) {
//         setError("Please fill in all required fields");
//         setLoading(false);
//         return;
//       }
//       if (username.length < 3) {
//         setError("Username must be at least 3 characters");
//         setLoading(false);
//         return;
//       }
//       if (password.length < 6) {
//         setError("Password must be at least 6 characters");
//         setLoading(false);
//         return;
//       }

//       const pwHash = await hashPassword(password);
      
//       // Call SpacetimeDB reducer
//       reducers.register_user(username, pwHash, role, displayName, role === "doctor" ? specialization : "");
      
//     } catch (err) {
//       setError(err.message || "Signup failed");
//       setLoading(false);
//     }
//   }

//   return (
//     <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center px-4 relative overflow-hidden">
//       {/* Decorative gradient orbs */}
//       <div className="absolute top-[-200px] right-[-200px] w-[500px] h-[500px] bg-gradient-to-br from-purple-600/20 to-pink-600/20 rounded-full blur-3xl" />
//       <div className="absolute bottom-[-200px] left-[-200px] w-[500px] h-[500px] bg-gradient-to-br from-blue-600/20 to-cyan-600/20 rounded-full blur-3xl" />

//       <div className="relative z-10 w-full max-w-md">
//         {/* Logo */}
//         <div className="text-center mb-8">
//           <Link to="/" className="inline-flex items-center gap-2">
//             <span className="text-xl text-white">✦</span>
//             <span className="text-white font-bold text-2xl tracking-wide font-syne">MedSpatial AI</span>
//           </Link>
//           <p className="text-gray-500 mt-2 text-sm">Create your account</p>
//         </div>

//         {/* Card */}
//         <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-8 shadow-2xl">
//           <form onSubmit={handleSubmit} className="space-y-4">
//             {error && (
//               <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm">
//                 {error}
//               </div>
//             )}

//             {/* Role Selector */}
//             <div>
//               <label className="block text-gray-400 text-sm mb-2 font-medium">I am a</label>
//               <div className="grid grid-cols-2 gap-3">
//                 <button
//                   type="button"
//                   id="role-patient"
//                   onClick={() => setRole("patient")}
//                   className={`py-3 rounded-xl text-sm font-semibold transition-all duration-300 border ${
//                     role === "patient"
//                       ? "bg-blue-600/20 border-blue-500/50 text-blue-400 shadow-lg shadow-blue-600/10"
//                       : "bg-white/5 border-white/10 text-gray-500 hover:border-white/20"
//                   }`}
//                 >
//                   🏥 Patient
//                 </button>
//                 <button
//                   type="button"
//                   id="role-doctor"
//                   onClick={() => setRole("doctor")}
//                   className={`py-3 rounded-xl text-sm font-semibold transition-all duration-300 border ${
//                     role === "doctor"
//                       ? "bg-purple-600/20 border-purple-500/50 text-purple-400 shadow-lg shadow-purple-600/10"
//                       : "bg-white/5 border-white/10 text-gray-500 hover:border-white/20"
//                   }`}
//                 >
//                   👨‍⚕️ Doctor
//                 </button>
//               </div>
//             </div>

//             {/* Display Name */}
//             <div>
//               <label className="block text-gray-400 text-sm mb-2 font-medium">Display Name</label>
//               <input
//                 type="text"
//                 id="signup-displayname"
//                 value={displayName}
//                 onChange={(e) => setDisplayName(e.target.value)}
//                 placeholder={role === "doctor" ? "Dr. Smith" : "Your name"}
//                 className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-600 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/25 transition"
//               />
//             </div>

//             {/* Username */}
//             <div>
//               <label className="block text-gray-400 text-sm mb-2 font-medium">Username</label>
//               <input
//                 type="text"
//                 id="signup-username"
//                 value={username}
//                 onChange={(e) => setUsername(e.target.value)}
//                 placeholder="Choose a username"
//                 className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-600 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/25 transition"
//               />
//             </div>

//             {/* Password */}
//             <div>
//               <label className="block text-gray-400 text-sm mb-2 font-medium">Password</label>
//               <input
//                 type="password"
//                 id="signup-password"
//                 value={password}
//                 onChange={(e) => setPassword(e.target.value)}
//                 placeholder="At least 6 characters"
//                 className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-600 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/25 transition"
//               />
//             </div>

//             {/* Specialization (Doctor only) */}
//             {role === "doctor" && (
//               <div className="animate-in">
//                 <label className="block text-gray-400 text-sm mb-2 font-medium">Specialization</label>
//                 <input
//                   type="text"
//                   id="signup-specialization"
//                   value={specialization}
//                   onChange={(e) => setSpecialization(e.target.value)}
//                   placeholder="e.g., Radiology, Pulmonology"
//                   className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-600 focus:outline-none focus:border-purple-500/50 focus:ring-1 focus:ring-purple-500/25 transition"
//                 />
//               </div>
//             )}

//             {/* Submit */}
//             <button
//               type="submit"
//               id="signup-submit"
//               disabled={loading}
//               className={`w-full py-3 rounded-xl font-semibold transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg ${
//                 role === "doctor"
//                   ? "bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 shadow-purple-600/25"
//                   : "bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 shadow-blue-600/25"
//               } text-white`}
//             >
//               {loading ? (
//                 <span className="flex items-center justify-center gap-2">
//                   <svg className="animate-spin w-5 h-5" viewBox="0 0 24 24">
//                     <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
//                     <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z" />
//                   </svg>
//                   Creating account...
//                 </span>
//               ) : (
//                 `Create ${role === "doctor" ? "Doctor" : "Patient"} Account`
//               )}
//             </button>
//           </form>

//           {/* Divider */}
//           <div className="mt-6 flex items-center gap-4">
//             <div className="flex-1 h-px bg-white/10" />
//             <span className="text-gray-600 text-xs">OR</span>
//             <div className="flex-1 h-px bg-white/10" />
//           </div>

//           {/* Login link */}
//           <p className="text-center mt-6 text-gray-500 text-sm">
//             Already have an account?{" "}
//             <Link to="/login" className="text-blue-400 hover:text-blue-300 font-medium transition">
//               Sign in
//             </Link>
//           </p>
//         </div>
//       </div>
//     </div>
//   );
// }
