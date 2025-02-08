import { useState, useEffect } from "react";
import { useRouter } from "next/router";
import Navbar from "../components/Navbar";

export default function Upload() {
  const router = useRouter();
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState("");
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [imageUrl, setImageUrl] = useState(""); // To store the image URL
  const [showModal, setShowModal] = useState(false); // To control the modal visibility
  const publicImages = {
    chu: ["/chu/chuu1.jpg", "/chu/chuu2.jpg", "/chu/chuu3.jpg", "/chu/chuu4.jpg"],
    cha: ["/cha/chaeunwoo1.jpg", "/cha/chaeunwoo2.jpg", "/cha/chaeunwoo3.jpg", "/cha/chaeunwoo4.jpg"],
    byeon: ["/byeon/byeon_1.png", "/byeon/byeon_2.png", "/byeon/byeon_3.png", "/byeon/byeon_4.png"],
    winter: ["/winter/winter1.jpg", "/winter/winter2.jpg", "winter/winter3.jpg", "winter/winter4.jpg"],
  };

  const [activeTab, setActiveTab] = useState("chu");


  const handlePublicImageSelect = (img) => {
    setImagePreview(img);
    // Fetch the image as a blob and set it as the image file
    fetch(img)
      .then((response) => response.blob())
      .then((blob) => {
      const file = new File([blob], "publicImage.jpg", { type: blob.type });
      setImage(file);
      })
      .catch((error) => console.error("Error fetching image:", error));
    setShowModal(false);
  };
  

  // Handle image upload
  useEffect(() => {
    if(!image) return;
    uploadImage();
  }, [image]);

 
  // Upload image and get the image URL
  const uploadImage = async () => {
    if (!image) {
      alert("Please upload an image first.");
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append("image", image);

      const response = await fetch("http://localhost:8000/upload-image", {
        method: "POST",
        body: image,
      });

      if (!response.ok) {
        throw new Error("Failed to upload image.");
      }

      const data = await response.json(); // Get the response data
      setImageUrl(data.image_url); // Set the image URL returned from the server
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("Failed to upload image. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Submit post with title, content, and image URL
  const submitPost = async () => {
    if (!imageUrl || !title || !content) {
      alert("Please fill in all fields and upload an image.");
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch("http://localhost:8000/upload-post", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user: activeTab,
          title,
          content,
          image_url: imageUrl, // Use the image URL obtained from image upload
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to upload post.");
      }

      // Redirect to home page after successful post upload
      alert("🔒 Image Encoded base on User info!")
      router.push("/");
    } catch (error) {
      console.error("Error uploading post:", error);
      alert("Failed to upload post. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      {/* Outer Container with Gradient Border */}
      
      <div className="w-full h-full bg-gradient-to-r from-teal-400 via-pink-500 to-yellow-500 p-3">
        {/* Inner Box */}
        <div className="flex h-full bg-neutral-100">
          {/* Main Content */}
          <Navbar />
          <div className="flex-1 p-8">
            <h1 className="text-5xl font-bold text-pink-500">Upload Post</h1>
            <p className="mt-4 font-bold text-black">
              Upload an image and add a title and content for your post.
            </p>


            <div className="mt-6">
              <input
                type="text"
                placeholder="Title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className="w-full p-3 pl-5 rounded-full border border-gray-300 shadow-md"
              />
            </div>

            <div className="mt-6 w-64 h-64 rounded-lg bg-gray-400">
              {!imagePreview && (
                <div className="w-full h-full rounded-lg">
                  <button
                    className="bg-orange-300 text-white w-full rounded-lg h-full shadow text-2xl hover:bg-orange-200"
                    onClick={() => setShowModal(true)}
                  >
                    <span className="material-icons text-white text-3xl">add</span>
                  </button>
                </div>
              )}

              {showModal && (
                <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
                  <div className="relative bg-white p-4 mx-[10vw] rounded-lg shadow-lg">
                    <h2 className="text-xl font-semibold mb-4">Select an image.</h2>
                    <div className="flex justify-center mb-4">
                      {Object.keys(publicImages).map((tab) => (
                        <button
                          key={tab}
                          className={`px-4 py-2 mx-2 rounded ${activeTab === tab ? "bg-gray-500 text-white scale-110 duration-300 transform" : "bg-gray-200"}`}
                          onClick={() => setActiveTab(tab)}
                        >
                          {tab}
                        </button>
                      ))}
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      {publicImages[activeTab].map((img, index) => (
                        <img
                          key={index}
                          src={img}
                          alt={`Public Image ${index}`}
                          className="w-full aspect-square object-cover cursor-pointer hover:scale-105 duration-300"
                          onClick={() => handlePublicImageSelect(img)}
                        />
                      ))}
                    </div>
                    <button
                      className="mt-4 absolute top-0 right-5 text-red-500 px-4 py-2 rounded hover:text-rose-700 hover:scale-105 duration-300"
                      onClick={() => setShowModal(false)}
                    >
                      Close
                    </button>
                  </div>
                </div>
              )}

              {imagePreview && (
                <div className="w-full h-full relative rounded-lg">
                  {/* X 버튼 */}
                <button
                  className="absolute top-2 right-2 p-2 text-rose-500 rounded-lg hover:text-rose-700 focus:outline-none"
                  onClick={() => setImagePreview(null)}
                >
                  ✕
                </button>
                <img
                  src={imagePreview}
                  alt="Uploaded Preview"
                  className="w-full h-full object-cover border-2 border-orange-400"
                />
              </div>
            )}
          </div>

            {/* Content Input */}
            <div className="mt-6">
              <textarea
                placeholder="Content"
                value={content}
                onChange={(e) => setContent(e.target.value)}
                className="w-full p-3 rounded border border-gray-300 shadow-md h-32"
              />
            </div>

            {/* Submit Post Button */}
            {imageUrl && content && title && (
              <div className="mt-6">
                <button
                  className="fixed bottom-[15vh] right-[5vw] bg-teal-400 text-white px-2 rounded-full shadow-lg hover:bg-teal-600 flex items-center justify-center"
                  onClick={submitPost}
                >
                  <span className="material-icons text-white text-3xl">check</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
