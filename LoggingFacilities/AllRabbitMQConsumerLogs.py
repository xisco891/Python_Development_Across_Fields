#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:16:17 2019

@author: francisco.vicente
"""

import logging
import pika

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')

LOGGER = logging.getLogger(__name__)


class ExampleConsumer(object):
    """This is an example consumer that will handle unexpected interactions
    with RabbitMQ such as channel and connection closures.

    If RabbitMQ closes the connection, it will reopen it. You should
    look at the output, as there are limited reasons why the connection may
    be closed, which usually are tied to permission related issues or
    socket timeouts.

    If the channel is closed, it will indicate a problem with one of the
    commands that were issued and that should surface in the output as well.

    """
    EXCHANGE = 'message'
    EXCHANGE_TYPE = 'topic'
    QUEUE = 'text'
    ROUTING_KEY = 'example.text'

    def __init__(self, amqp_url):
        """Create a new instance of the consumer class, passing in the AMQP
        URL used to connect to RabbitMQ.

        :param str amqp_url: The AMQP url to connect with

        """
        self._connection = None
        self._channel = None
        self._closing = False
        self._consumer_tag = None
        self._url = amqp_url

    def connect(self):
        """This method connects to RabbitMQ, returning the connection handle.
        When the connection is established, the on_connection_open method
        will be invoked by pika.

        :rtype: pika.SelectConnection

        """
        LOGGER.info('Connecting to %s', self._url)
        return pika.SelectConnection(pika.URLParameters(self._url),
                                     self.on_connection_open,
                                     stop_ioloop_on_close=False)

    def on_connection_open(self, unused_connection):
        """This method is called by pika once the connection to RabbitMQ has
        been established. It passes the handle to the connection object in
        case we need it, but in this case, we'll just mark it unused.

        :type unused_connection: pika.SelectConnection

        """
        LOGGER.info('Connection opened')
        self.add_on_connection_close_callback()
        self.open_channel()

    def add_on_connection_close_callback(self):
        """This method adds an on close callback that will be invoked by pika
        when RabbitMQ closes the connection to the publisher unexpectedly.

        """
        LOGGER.info('Adding connection close callback')
        self._connection.add_on_close_callback(self.on_connection_closed)

    def on_connection_closed(self, connection, reply_code, reply_text):
        """This method is invoked by pika when the connection to RabbitMQ is
        closed unexpectedly. Since it is unexpected, we will reconnect to
        RabbitMQ if it disconnects.

        :param pika.connection.Connection connection: The closed connection obj
        :param int reply_code: The server provided reply_code if given
        :param str reply_text: The server provided reply_text if given

        """
        self._channel = None
        if self._closing:
            self._connection.ioloop.stop()
        else:
            LOGGER.warning('Connection closed, reopening in 5 seconds: (%s) %s',
                           reply_code, reply_text)
            self._connection.add_timeout(5, self.reconnect)

    def reconnect(self):
        """Will be invoked by the IOLoop timer if the connection is
        closed. See the on_connection_closed method.

        """
        # This is the old connection IOLoop instance, stop its ioloop
        self._connection.ioloop.stop()

        if not self._closing:

            # Create a new connection
            self._connection = self.connect()

            # There is now a new connection, needs a new ioloop to run
            self._connection.ioloop.start()

    def open_channel(self):
        LOGGER.info('Creating a new channel')
   
    def on_channel_open(self, channel):
        LOGGER.info('Channel opened')
   
    def add_on_channel_close_callback(self):
        LOGGER.info('Adding channel close callback')
   
    def on_channel_closed(self, channel, reply_code, reply_text):
     
        LOGGER.warning('Channel %i was closed: (%s) %s',
                       channel, reply_code, reply_text)
   
    def setup_exchange(self, exchange_name):
        LOGGER.info('Declaring exchange %s', exchange_name)

    def on_exchange_declareok(self, unused_frame):
        LOGGER.info('Exchange declared')

    def setup_queue(self, queue_name):
        LOGGER.info('Declaring queue %s', queue_name)

    def on_queue_declareok(self, method_frame):
        LOGGER.info('Binding %s to %s with %s',

    def on_bindok(self, unused_frame):
        LOGGER.info('Queue bound')

    def start_consuming(self):
        LOGGER.info('Issuing consumer related RPC commands')

    def add_on_cancel_callback(self):
        LOGGER.info('Adding consumer cancellation callback')
    def on_consumer_cancelled(self, method_frame):
        LOGGER.info('Consumer was cancelled remotely, shutting down: %r',

    def on_message(self, unused_channel, basic_deliver, properties, body):
        LOGGER.info('Received message # %s from %s: %s',

    def acknowledge_message(self, delivery_tag):
        LOGGER.info('Acknowledging message %s', delivery_tag)

    def stop_consuming(self):
        LOGGER.info('Sending a Basic.Cancel RPC command to RabbitMQ')
    def on_cancelok(self, unused_frame):
        LOGGER.info('RabbitMQ acknowledged the cancellation of the consumer')

    def close_channel(self):
        LOGGER.info('Closing the channel')

    def stop(self):     
        LOGGER.info('Stopping')
        LOGGER.info('Stopped')

    def close_connection(self):
        LOGGER.info('Closing connection')
        

def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    
    try:
        example.run()
    except KeyboardInterrupt:
        example.stop()


if __name__ == '__main__':
    main()