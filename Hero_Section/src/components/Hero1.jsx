import { useEffect, useRef } from "react";

export default function Hero1() {
  const boxRef = useRef(null);

  useEffect(() => {
    let ticking = false;

    const handleScroll = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          const scrollY = window.scrollY;

          const progress = Math.min(scrollY / 600, 1);
          const ease = 1 - Math.pow(1 - progress, 3);

          const scale = 1 - 0.9 * ease;
          const radius = 50 * ease;

          if (boxRef.current) {
            boxRef.current.style.transform = `scale(${scale})`;
            boxRef.current.style.borderRadius = `${radius}%`;
          }

          ticking = false;
        });

        ticking = true;
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div className="absolute inset-0 z-0 pointer-events-none">
      <div
        ref={boxRef}
        className="w-full h-full overflow-hidden will-change-transform"
      >
        <video
          src="/neuron.mp4"
          autoPlay
          muted
          loop
          playsInline
          className="w-full h-full object-cover object-center"
        />
      </div>
    </div>
  );
}