import Hero1 from "./Hero1.jsx";
// import RotatingModel from "./DetailSection.jsx";
import DetailSection from "./DetailSection";
import { useEffect } from "react";
import { useLocation } from "react-router-dom";

export default function Home() {
  const location = useLocation();

  useEffect(() => {
    if (location.hash) {
      // delay ensures DOM + 3D canvas loads first
      setTimeout(() => {
        const el = document.querySelector(location.hash);
        if (el) {
          el.scrollIntoView({ behavior: "smooth" });
        }
      }, 150);
    }
  }, [location]);

  return (
    <>
      {/* HERO SECTION */}
      <div className="relative h-screen w-full overflow-hidden">
        <Hero1 />

        <div className="relative z-10 h-full w-full flex items-center justify-center">
          <h1 className="text-[#f5f5f5] text-[55px] absolute font-syne left-20 bottom-20 font-bold">
            Seeing Beyond Scans.<br />
            Saving Lives.
          </h1>
        </div>
      </div>

      {/* OTHER SECTIONS */}
      <DetailSection />
    </>
  );
}