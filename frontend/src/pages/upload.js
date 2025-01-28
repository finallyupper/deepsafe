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

  // Handle image upload
  useEffect(() => {
    if(!image) return;
    uploadImage();
  }, [image]);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setImagePreview(URL.createObjectURL(file)); // Show image preview
    }
  };

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
          title,
          content,
          image_url: imageUrl, // Use the image URL obtained from image upload
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to upload post.");
      }

      // Redirect to home page after successful post upload
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

            {/* Title Input */}
            <div className="mt-6">
              <input
                type="text"
                placeholder="Title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className="w-full p-3 pl-5 rounded-full border border-gray-300 shadow-md"
              />
            </div>

            {/* Image Upload */}
            <div className="mt-6 w-64 h-64 rounded-lg bg-gray-400">
              {!imagePreview && (
                <div className="w-full h-full rounded-lg">
                  <button
                    className="bg-orange-300 text-white w-full rounded-lg h-full shadow text-2xl hover:bg-orange-200"
                    onClick={() => document.getElementById("image-upload").click()}
                  >
                    <span className="material-icons text-white text-3xl">add</span>
                  </button>
                  <input
                    id="image-upload"
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleImageUpload}
                  />
                </div>
              )}
              {/* Image Preview */}
              {imagePreview && (
                <div className="w-full h-full relative rounded-lg">
                  {/* X 버튼 */}
                  <button
                    className="absolute top-4 right-4 p-2 text-gray-500 rounded-lg hover:text-gray-700 focus:outline-none"
                    onClick={() => setImagePreview(null)}
                  >
                    ✕
                  </button>
                  <img
                    src={imagePreview}
                    alt="Uploaded Preview"
                    className="mt-4 w-full h-full object-cover border-2 border-orange-400"
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
