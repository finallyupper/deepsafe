import { useState, useEffect } from "react";
import { useRouter } from 'next/router';

import Navbar from "../../components/Navbar";



export default function Home() {
  const router = useRouter();

  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [swappedImage, setSwappedImage] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const [showModal, setShowModal] = useState(false); // To control the modal visibility
  const [posts, setPosts] = useState([]);
  const [showPostModal, setShowPostModal] = useState(false);
  const [selectedPost, setSelectedPost] = useState("");
  const [imageUser, setImageUser] = useState("");
  
  const [publicImages, setPublicImages] = useState({});

  useEffect(() => {
    fetch("/api/images")
      .then((res) => res.json())
      .then((data) => setPublicImages(data))
      .catch((err) => console.error(err));
  }, []);

  
  useEffect(() => {
    if(!image) return;
    uploadImage();
  }, [image]);



  const [activeTab, setActiveTab] = useState("chu");

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

  
  // 페이스스왑 요청 핸들러
  const faceswap = async () => {
    if (!image) {
      alert("Please upload an image first!");
      return;
    }

    setIsLoading(true);
    try {
      const body = {
        target_image_url: imageUrl,
        source_image_url: selectedPost,
      };

      // API 요청
      const response = await fetch("http://localhost:8000/face-swap", {
        method: "POST",
        body: JSON.stringify(body),
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error("Failed to process the image.");
      }

      const data = await response.json(); // 서버로부터 데이터 받기
      setSwappedImage(data.swappedImageUrl); // 워터마킹된 이미지 URL 저장
      alert(data.message); // 성공 메시지
    } catch (error) {
      console.error("Error during watermarking:", error);
      alert("Failed to watermark the image. Please try again.");
    } finally {
      setIsLoading(false); // 로딩 종료
    }
  };

  const matchPair = (user) => {
    switch (user) {
      case "chu":
        return "win";
      case "cha":
        return "byeon";
      case "byeon":
        return "cha";
      case "win":
        return "chu";
      default:
        return "Unknown";
    }
  }

  // 로딩 스피너 컴포넌트
  const LoadingSpinner = () => {
    return (
      <div className="flex items-center justify-center mt-2">
        <div className="w-8 h-8 border-4 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      {/* Outer Container with Gradient Border */}
      <div className="w-full h-full bg-red-300 p-3">
        {/* Inner Box */}
        {isLoading && (
          <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
            <div className="flex flex-col items-center bg-white p-6 rounded-lg shadow-lg">
              <div className="w-10 h-10 border-4 border-t-transparent border-rose-500 rounded-full animate-spin"></div>
              <p className="mt-4 text-lg font-semibold text-gray-700">Face Swapping...</p>
            </div>
          </div>
        )}
        <div className="flex h-full bg-neutral-100">
          {/* Navigation Bar */}
          <Navbar />

          {/* Main Content */}
          <div className="flex-1 p-8">
            <h1 className="text-5xl font-bold text-red-500">
              FaceSwap
            </h1>
            <p className="mt-4 text-black font-semibold">
              Upload images to swap faces. Watermarked image will not be swapped well.
            </p>
             {/* Image Upload */}
             <p className="mt-12 font-semibold">Target Image</p>
             <div className="w-64 h-64 rounded-lg bg-gray-400">
              
              {!imagePreview && (
                <div className="w-full h-full rounded-lg">
                  <button
                    className="bg-rose-300 text-white w-full rounded-lg h-full shadow text-2xl hover:bg-rose-200"
                    onClick={() => setShowModal(true)}
                  >
                    <span className="material-icons text-white text-3xl">image</span>
                  </button>
                </div>
              )}

              {showModal && (
                <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
                  <div className="relative bg-white p-4 mx-[10vw] rounded-lg shadow-lg">
                    <h2 className="text-xl font-semibold mb-4">Select an target(background) image.</h2>
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
                    <div className="grid grid-cols-4 gap-4">
                      {publicImages[activeTab].map((img, index) => (
                        <img
                          key={index}
                          src={img}
                          alt={`Public Image ${index}`}
                          className="w-48 aspect-square object-cover cursor-pointer hover:scale-105 duration-300"
                          onClick={() => {handlePublicImageSelect(img); setImageUser(activeTab)}}
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
              {/* Image Preview */}
              {imagePreview && (
                <div className="w-full h-full relative rounded-lg">
                  
                  <button
                    className="absolute top-2 right-2 p-2 text-rose-500 rounded-lg hover:text-rose-700 focus:outline-none"
                    onClick={() => setImagePreview(null)}
                  >
                    ✕
                  </button>
                  <img
                    src={imagePreview}
                    alt="Uploaded Preview"
                    className="w-full h-full object-cover border-2 border-rose-400"
                  />
                  </div>
                  )}

                  </div>
                  <p className="mt-12 font-semibold">Source Image</p>
                  <div className="w-64 h-64 rounded-lg bg-gray-400">
                    {!selectedPost && (
                      <div className="w-full h-full rounded-lg">
                        <button
                          className="bg-rose-300 text-white w-full rounded-lg h-full shadow text-2xl hover:bg-rose-200"
                          onClick={() => setShowPostModal(true)}
                        >
                          <span className="material-icons text-white text-3xl">person</span>
                        </button>
                      </div>
                    )}
                     {showPostModal && (
                      <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
                        <div className="relative bg-white p-4 mx-[10vw] rounded-lg shadow-lg">
                          <h2 className="text-xl text-black font-semibold mb-4">Select a source(face) post image.</h2>
                          <div className="grid grid-cols-4 gap-2 mt-8">
                            {posts.filter(post => post.user === matchPair(imageUser)).map((post) => (
                              <div
                                key={post.id}
                                className="relative cursor-pointer h-24 aspect-square hover:scale-105 duration-300 bg-gray-200 shadow-md overflow-hidden"
                                onClick={() => {
                                  setSelectedPost(post.image_url);
                                  setShowPostModal(false);
                                }}
                              >
                                <img
                                  src={`http://localhost:8000${post.image_url}`}
                                  alt={post.title}
                                  className="w-full aspect-square object-cover"
                                />
                              </div>
                            ))}
                          </div>
                          <button
                            className="mt-4 absolute top-0 right-5 text-red-500 px-4 py-2 rounded hover:text-rose-700 hover:scale-105 duration-300"
                            onClick={() => {setShowPostModal(false); setImageUser("")}}
                          >
                            Close
                          </button>
                        </div>
                      </div>
                    )}

                    {selectedPost && (
                      <div className="w-full h-full relative rounded-lg">
                        {/* X 버튼 */}
                  <button
                    className="absolute top-2 right-2 p-2 text-rose-500 rounded-lg hover:text-rose-700 focus:outline-none"
                    onClick={() => setSelectedPost(null)}
                  >
                    ✕
                  </button>
                  <img
                    src={`http://localhost:8000${selectedPost}`}
                    alt="Uploaded Preview"
                    className="w-full h-full object-cover border-2 border-rose-400"
                  />
                </div>
              )}

     
             
               {imageUrl && selectedPost && (
                  <div className="mt-6">
                    <button
                      className="fixed bottom-[15vh] right-[5vw] bg-rose-400 text-white px-2 py-1 rounded-full shadow-lg hover:bg-rose-600 flex items-center justify-center"
                      onClick={faceswap}
                    >
                      <span className="material-icons text-white-500 text-3xl">sentiment_very_dissatisfied</span>
                    </button>
                  </div>
                )}
            </div>
           
          </div>
          {swappedImage && (
          <div className="fixed w-64 h-64 bottom-[40vh] right-[25vw]">
            <h2 className="text-2xl font-bold text-rose-500">Result</h2>
            {isLoading ? (
              <LoadingSpinner />
            ) : (
              swappedImage && (
                <img
                  src={`http://localhost:8000${swappedImage}`}
                  alt="Swapped Result"
                  className="w-full h-full object-cover border-2 border-rose-400 mt-4"
                />
              )
            )}
          </div>)}
        </div>
        
      </div>
    </div>
  );
}
