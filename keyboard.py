'''Code retrieved from:
http://vjethava.blogspot.com/2010/11/matlabs-keyboard-command-in-python.html'''
import code
import sys

def keyboard(banner=None):
	''' Function that mimics the matlab keyboard command '''
	# use exception trick to pick up the current frame
	try:
		raise None
	except:
		frame = sys.exc_info()[2].tb_frame.f_back
	if sys.version_info < (3,):
		eval('print "# Use quit() to exit :) Happy debugging!"')
	else:
		eval('print("# Use quit() to exit :) Happy debugging!")')
	# evaluate commands in current namespace
	namespace = frame.f_globals.copy()
	namespace.update(frame.f_locals)
	if not ('quit' in namespace): 		# Make sure that 'quit' is defined
		def leave():
			raise SystemExit	
		namespace['quit']=leave
	
	def exitNLevel(n=1):
		NLevel = 'raise SystemExit; ' * n
		exec(NLevel)
	namespace['exitNLevel']=exitNLevel
	
	try:
		code.interact(banner=banner, local=namespace)
	except SystemExit:
		return 