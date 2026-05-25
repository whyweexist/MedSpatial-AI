// import React, { createContext, useContext, useState, useEffect } from "react";
// import { connectToSpacetimeDB } from "../lib/spacetimedb";
// import { tables, reducers } from "../module_bindings";

// const AuthContext = createContext(null);

// export function AuthProvider({ children }) {
//   const [user, setUser] = useState(null);
//   const [isAuthenticated, setIsAuthenticated] = useState(false);
//   const [isDBConnected, setIsDBConnected] = useState(false);

//   useEffect(() => {
//     // 1. Get previous token if any
//     const savedToken = localStorage.getItem("medspatial_stdb_token");

//     // 2. Connect to SpacetimeDB
//     const conn = connectToSpacetimeDB(savedToken);

//     // 3. Define event handlers
//     const unsubConnect = conn.onConnect((c, identity, token) => {
//       setIsDBConnected(true);
//       // Wait for table sync; a user record matching our identity means we're logged in
//     });

//     const unsubDisconnect = conn.onDisconnect(() => {
//       setIsDBConnected(false);
//     });

//     // 4. Handle table inserts/updates
//     function checkIdentityUser() {
//       const myIdentity = conn.identity();
//       if (!myIdentity) return;

//       const myUser = conn.user.identity.find(myIdentity);
//       if (myUser) {
//         setUser(myUser);
//         setIsAuthenticated(true);
//       }
//     }

//     const unsubUserInsert = conn.user.onInsert(() => checkIdentityUser());
//     const unsubUserUpdate = conn.user.onUpdate(() => checkIdentityUser());

//     // Fallback interval to check after initial sync
//     const interval = setInterval(() => checkIdentityUser(), 1000);

//     return () => {
//       // unsubConnect(); // if you had the remove logic
//       // unsubDisconnect();
//       clearInterval(interval);
//     };
//   }, []);

//   const login = (userData) => {
//     // This frontend "login" sets state immediately if needed, but usually we just
//     // rely on stdb table updates.
//     setUser(userData);
//     setIsAuthenticated(true);
//   };

//   const signup = (userData) => {
//     setUser(userData);
//     setIsAuthenticated(true);
//   };

//   const logout = () => {
//     setUser(null);
//     setIsAuthenticated(false);
//     localStorage.removeItem("medspatial_stdb_token");
//     window.location.href = "/login"; // Force reload to get a new identity
//   };

//   return (
//     <AuthContext.Provider value={{ user, isAuthenticated, isDBConnected, login, signup, logout }}>
//       {children}
//     </AuthContext.Provider>
//   );
// }

// export function useAuth() {
//   const ctx = useContext(AuthContext);
//   if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
//   return ctx;
// }
