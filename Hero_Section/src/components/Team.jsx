const team = [
  {
    name: "Dhruv Tiwari",
    role: "AI Research Lead",
    desc: "Focused on building intelligent healthcare models and diagnostic systems.",
    img: "https://randomuser.me/api/portraits/men/32.jpg",
  },
  {
    name: "Yash Rajput",
    role: "3D Visualization Engineer",
    desc: "Transforms medical scans into interactive 3D visual experiences.",
    img: "https://randomuser.me/api/portraits/women/44.jpg",
  },
  {
    name: "Abhay Pandit",
    role: "Backend Developer",
    desc: "Designs scalable systems for processing complex medical data.",
    img: "https://randomuser.me/api/portraits/men/65.jpg",
  },
  {
    name: "Mahima Patel",
    role: "UI/UX Designer",
    desc: "Crafts intuitive interfaces for better healthcare interaction.",
    img: "https://randomuser.me/api/portraits/women/68.jpg",
  },
];

export default function Team() {
  return (
    <section className="bg-gray-50 py-16 mt-10 px-6 md:px-20">
      <div className="max-w-7xl mx-auto">
        
        {/* Heading Section */}
        <div className="grid md:grid-cols-3 gap-10 mb-16">
          <h2 className="text-4xl md:text-5xl font-bold leading-tight">
            We Are <br /> Detail <br /> Oriented
          </h2>

          <p className="text-gray-600 text-lg">
            Our team converts complex ideas into simple, interactive experiences.
            We focus on precision, innovation, and usability to create impactful
            solutions.
          </p>

          <p className="text-gray-600 text-lg">
            Powered by cutting-edge technology, we collaborate to deliver
            meaningful insights and high-quality digital products.
          </p>
        </div>

        {/* Team Grid */}
        <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-8">
          {team.map((member, index) => (
            <div
              key={index}
              className="bg-white rounded-2xl shadow-sm hover:shadow-lg transition p-6 text-center"
            >
              <img
                src={member.img}
                alt={member.name}
                className="w-24 h-24 mx-auto rounded-full object-cover mb-4"
              />
              <h3 className="text-xl font-semibold">{member.name}</h3>
              <p className="text-sm text-blue-600 mb-2">{member.role}</p>
              <p className="text-gray-500 text-sm">{member.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}