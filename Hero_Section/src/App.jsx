import { Routes, Route } from "react-router-dom";
import Layout from "./layout";
import Team from "./components/Team";
import PatientDashboard from "./Pages/PatientCommunity";

// Pages
import Home from "./components/Home";
// import PatientDashboard from "./Pages/PatientCommunity";
// import DoctorDashboard from "./Pages/DoctorCommunity";
// // import Login from "./Pages/Login";
// import Signup from "./Pages/Signup";

function App() {
  return (
    <Routes>


      {/* Public Layout Route */}
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="/team" element={<Team />} />
        <Route path="/patient" element={<PatientDashboard />} />
      </Route>
    </Routes>
  );
}

export default App;