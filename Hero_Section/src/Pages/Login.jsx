// import React, { useState, useEffect } from "react";
// import { Link, useNavigate } from "react-router-dom";
// import { useAuth } from "../context/AuthContext";
// import { reducers } from "../module_bindings";

// export default function Login() {
//   const [username, setUsername] = useState("");
//   const [password, setPassword] = useState("");
//   const [error, setError] = useState("");
//   const [loading, setLoading] = useState(false);
//   const { user } = useAuth();
//   const navigate = useNavigate();

//   // Listen to SpacetimeDB reducer callbacks
//   useEffect(() => {
//     const unsub = reducers.onLoginUser((ctx, username, hash) => {
//       setLoading(false);
//       if (ctx.event.status === "failed") {
//         setError(ctx.event.callerError || "Login failed");
//         return;
//       }
//       // On success, AuthContext will auto-catch the updated identity and set 'user'
//     });
//     return () => {
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
//       if (!username || !password) {
//         setError("Please fill in all fields");
//         setLoading(false);
//         return;
//       }

//       const pwHash = await hashPassword(password);
      
//       // Call SpacetimeDB reducer
//       reducers.login_user(username, pwHash);
      
//     } catch (err) {
//       setError(err.message || "Login failed");
//       setLoading(false);
//     }
//   }

//   return (
//     <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center px-4 relative overflow-hidden">
//       {/* Decorative gradient orbs */}
//       <div className="absolute top-[-200px] left-[-200px] w-[500px] h-[500px] bg-gradient-to-br from-blue-600/20 to-purple-600/20 rounded-full blur-3xl" />
//       <div className="absolute bottom-[-200px] right-[-200px] w-[500px] h-[500px] bg-gradient-to-br from-cyan-600/20 to-green-600/20 rounded-full blur-3xl" />

//       <div className="relative z-10 w-full max-w-md">
//         {/* Logo */}
//         <div className="text-center mb-8">
//           <Link to="/" className="inline-flex items-center gap-2">
//             <span className="text-xl text-white">✦</span>
//             <span className="text-white font-bold text-2xl tracking-wide font-syne">MedSpatial AI</span>
//           </Link>
//           <p className="text-gray-500 mt-2 text-sm">Sign in to your account</p>
//         </div>

//         {/* Card */}
//         <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-8 shadow-2xl">
//           <form onSubmit={handleSubmit} className="space-y-5">
//             {error && (
//               <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm">
//                 {error}
//               </div>
//             )}

//             {/* Username */}
//             <div>
//               <label className="block text-gray-400 text-sm mb-2 font-medium">Username</label>
//               <input
//                 type="text"
//                 id="login-username"
//                 value={username}
//                 onChange={(e) => setUsername(e.target.value)}
//                 placeholder="Enter your username"
//                 className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-600 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/25 transition"
//               />
//             </div>

//             {/* Password */}
//             <div>
//               <label className="block text-gray-400 text-sm mb-2 font-medium">Password</label>
//               <input
//                 type="password"
//                 id="login-password"
//                 value={password}
//                 onChange={(e) => setPassword(e.target.value)}
//                 placeholder="Enter your password"
//                 className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-600 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/25 transition"
//               />
//             </div>

//             {/* Submit */}
//             <button
//               type="submit"
//               id="login-submit"
//               disabled={loading}
//               className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-xl font-semibold hover:from-blue-500 hover:to-purple-500 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-600/25"
//             >
//               {loading ? (
//                 <span className="flex items-center justify-center gap-2">
//                   <svg className="animate-spin w-5 h-5" viewBox="0 0 24 24">
//                     <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
//                     <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z" />
//                   </svg>
//                   Signing in...
//                 </span>
//               ) : (
//                 "Sign In"
//               )}
//             </button>
//           </form>

//           {/* Divider */}
//           <div className="mt-6 flex items-center gap-4">
//             <div className="flex-1 h-px bg-white/10" />
//             <span className="text-gray-600 text-xs">OR</span>
//             <div className="flex-1 h-px bg-white/10" />
//           </div>

//           {/* Sign up link */}
//           <p className="text-center mt-6 text-gray-500 text-sm">
//             Don't have an account?{" "}
//             <Link to="/signup" className="text-blue-400 hover:text-blue-300 font-medium transition">
//               Create one
//             </Link>
//           </p>
//         </div>
//       </div>
//     </div>
//   );
// }
