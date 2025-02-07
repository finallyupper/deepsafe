import { useState, useEffect } from "react";
import { useRouter } from "next/router";
import Navbar from "../components/Navbar";

export default function Home() {
  const router = useRouter();
  const [posts, setPosts] = useState([]);
  const [selectedPost, setSelectedPost] = useState(null);

  useEffect(() => {
    // Fetch posts from the API
    const fetchPosts = async () => {
      try {
        const response = await fetch("http://localhost:8000/posts");
        if (!response.ok) {
          throw new Error("Failed to fetch posts");
        }
        const data = await response.json();
        setPosts(data);
      } catch (error) {
        console.error("Error fetching posts:", error);
      }
    };

    fetchPosts();
  }, []);

  const openPost = (post) => {
    setSelectedPost(post);
  };

  const closePost = () => {
    setSelectedPost(null);
  };

  return (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      {/* Outer Container with Gradient Border */}
      <div className="w-full h-full bg-blue-300 p-3">
        {/* Inner Box */}
        <div className="flex h-full bg-neutral-100">
          {/* Navigation Bar */}
          <Navbar />

          {/* Main Content */}
          <div className="flex-1 p-8">
            <h1 className="text-5xl font-bold text-blue-500">DeepSafe</h1>
            <p className="mt-4 font-semibold text-black">
              Protect face swaps. We will watermark the image if you upload some image.
            </p>

            {/* Upload Button */}
            <button
              className="fixed bottom-[15vh] right-[5vw] bg-blue-400 text-white px-2 rounded-full shadow-lg hover:bg-blue-600 flex items-center justify-center"
              onClick={() => router.push("/upload")}
            >
              <span className="material-icons text-white text-3xl">edit</span>
            </button>

            {/* Posts Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
              {posts.map((post) => (
                <div
                  key={post.id}
                  className="cursor-pointer hover:scale-105 duration-300 bg-gray-200 rounded-lg shadow-md overflow-hidden"
                  onClick={() => openPost(post)}
                >
                  <img
                    src={`http://localhost:8000${post.image_url}`} // 서버 URL과 결합
                    alt={post.title}
                    className="w-full h-48 object-cover"
                  />
                  <div className="p-2">
                    <h2 className="text-lg font-semibold text-gray-800 truncate">
                      {post.title}
                    </h2>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Modal for Selected Post */}
      {selectedPost && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="relative bg-white p-6 rounded-lg shadow-lg max-w-md w-full">
            {/* Close Button */}
            <button
              className="absolute top-4 right-4 p-2 text-gray-500 hover:text-gray-700 focus:outline-none"
              onClick={closePost}
            >
              ✕
            </button>

            {/* Image */}
            <img
              src={`http://localhost:8000${selectedPost.image_url}`} // 서버 URL과 결합
              alt={selectedPost.title}
              className="w-full h-64 object-cover rounded-lg"
            />

            {/* Title */}
            <h2 className="text-2xl font-bold text-gray-800 mt-4">
              {selectedPost.title}
            </h2>

            {/* Content */}
            <p className="text-gray-600 mt-2">{selectedPost.content}</p>
          </div>
        </div>
      )}
    </div>
  );
}
