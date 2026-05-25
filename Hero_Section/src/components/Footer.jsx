export default function Footer() {
  return (
    <footer className="bg-white border-t mt-20 px-6 md:px-20 py-12">
      <div className="max-w-7xl mx-auto grid md:grid-cols-4 gap-10">
        
        {/* Brand */}
        <div>
          <h2 className="text-2xl font-bold mb-4">C2THREE</h2>
          <p className="text-gray-600 text-sm">
            We transform medical data into meaningful insights using AI and
            3D visualization, helping improve healthcare decisions.
          </p>
        </div>

        {/* Navigation */}
        <div>
          <h3 className="font-semibold mb-4">Navigation</h3>
          <ul className="space-y-2 text-gray-600 text-sm">
            <li className="hover:text-black cursor-pointer">Vision</li>
            <li className="hover:text-black cursor-pointer">
              Research & Technology
            </li>
            <li className="hover:text-black cursor-pointer">Team</li>
          </ul>
        </div>

        {/* Resources */}
        <div>
          <h3 className="font-semibold mb-4">Resources</h3>
          <ul className="space-y-2 text-gray-600 text-sm">
            <li className="hover:text-black cursor-pointer">Documentation</li>
            <li className="hover:text-black cursor-pointer">Support</li>
            <li className="hover:text-black cursor-pointer">Privacy Policy</li>
          </ul>
        </div>

        {/* Subscribe */}
        <div>
          <h3 className="font-semibold mb-4">Stay Updated</h3>
          <p className="text-gray-600 text-sm mb-4">
            Subscribe to get the latest updates.
          </p>

          <div className="flex items-center border border-gray-300 rounded-full overflow-hidden w-full max-w-md">
  <input
    type="email"
    placeholder="Enter your email"
    className="flex-1 px-4 py-2 pr-2 text-sm outline-none"
  />
  <button className="bg-black text-white px-5 py-2 text-sm hover:bg-gray-800 transition">
    Subscribe
  </button>
</div>
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="border-t mt-10 pt-6 flex flex-col md:flex-row justify-between items-center text-sm text-gray-500">
        <p>© {new Date().getFullYear()} CBBR. All rights reserved.</p>

        <div className="flex gap-4 mt-4 md:mt-0">
          <span className="hover:text-black cursor-pointer">Twitter</span>
          <span className="hover:text-black cursor-pointer">LinkedIn</span>
          <span className="hover:text-black cursor-pointer">GitHub</span>
        </div>
      </div>
    </footer>
  );
}