export default function Home() {
  return (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      {/* Outer Container with Gradient Border */}
      <div className="w-3/4 h-3/4 bg-gradient-to-r from-teal-400 via-pink-500 to-yellow-500 p-1 rounded-lg">
        {/* Inner Box */}
        <div className="flex h-full bg-neutral-100 rounded-lg">
          {/* Navigation Bar */}
          <nav className="w-1/5 bg-gray-200 p-4 rounded-l-lg flex flex-col items-center">

            {/* Additional Navigation Items */}
            <button className="w-12 h-12 flex items-center justify-center mb-4 bg-gray-300 text-gray-700 rounded-full shadow-lg hover:bg-gray-400">
              <span className="material-icons">settings</span>
            </button>
            <button className="w-12 h-12 flex items-center justify-center bg-gray-300 text-gray-700 rounded-full shadow-lg hover:bg-gray-400">
              <span className="material-icons">info</span>
            </button>
          </nav>

          {/* Main Content */}
          <div className="flex-1 p-8">
            <h1 className="text-5xl font-bold text-blue-500">
              Hello, Tailwind CSS!
            </h1>
            <p className="mt-4 text-gray-600">
              This is a tropical-themed layout with a gradient border and a
              navigation bar on the left.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
