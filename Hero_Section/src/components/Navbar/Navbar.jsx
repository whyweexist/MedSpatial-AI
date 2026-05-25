import { Link } from "react-router-dom";
export default function Navbar() {
  return (
    <div className="fixed top-0 left-0 w-full flex justify-center z-50">
      {" "}
      <nav className="mt-4 w-[90%] max-w-6xl backdrop-blur-md bg-gray-200/30 border border-gray-300/30 rounded-full px-6 py-3 flex items-center justify-between shadow-lg">
        {" "}
        {/* Left Logo */}{" "}
        <div className="flex items-center gap-2 text-gray-800 font-semibold">
          {" "}
          <span className="text-lg">✦</span> <span>C2THREE</span>{" "}
        </div>{" "}
        {/* Center Links */}{" "}
        <div className="hidden md:flex gap-8 text-gray-700 text-sm tracking-wide">
  <Link to="/#vision" className="hover:text-gray-900 transition">
    VISION
  </Link>

  <Link to="/#features" className="hover:text-gray-900 transition">
    FEATURES
  </Link>

  <Link to="/patient" className="hover:text-gray-900 transition">
    COMMUNITY
  </Link>

  <Link to="/team" className="hover:text-gray-900 transition">
    TEAM
  </Link>

  <Link to="/#story" className="hover:text-gray-900 transition">
    OUR STORY
  </Link>
</div>
        {/* Right Button */}{" "}
        <div>
          {" "}
          <button className="bg-gray-900 text-white px-4 py-1.5 rounded-full text-sm hover:bg-black transition">
            {" "}
            TRY NOW !{" "}
          </button>{" "}
        </div>{" "}
      </nav>{" "}
    </div>
  );
}
