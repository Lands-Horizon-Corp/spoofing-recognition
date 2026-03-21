wrk.method = "POST"

-- Read your image file (make sure debug_image.jpg is in the same folder)
local file = io.open("debug_image.jpg", "rb")
local content = file:read("*all")
file:close()

-- Construct the multipart form-data payload
local boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
local body = "--" .. boundary .. "\r\n"
body = body .. "Content-Disposition: form-data; name=\"file\"; filename=\"debug_image.jpg\"\r\n"
body = body .. "Content-Type: image/jpeg\r\n\r\n"
body = body .. content .. "\r\n"
body = body .. "--" .. boundary .. "--\r\n"

wrk.body = body
wrk.headers["Content-Type"] = "multipart/form-data; boundary=" .. boundary
